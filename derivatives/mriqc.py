import os
import sys
import argparse
import bids
import subprocess
import json
import re
import pathlib
#from .fmriprep.fmriprep import load_bidsignore

script_dir = os.path.dirname(__file__)

PYBIDS_CACHE_PATH = ".pybids_cache"
SLURM_JOB_DIR = ".slurm"

MRIQC_REQ = {"cpus": 8, "mem_per_cpu": 4, "time": "8:00:00", "omp_nthreads": 8}

MRIQC_DEFAULT_VERSION = "mriqc-0.16"
MRIQC_DEFAULT_SINGULARITY_PATH = os.path.abspath(
    os.path.join(script_dir, f"../containers/{MRIQC_DEFAULT_VERSION}.sif")
)
SINGULARITY_CMD_BASE = " ".join(
    [
        "singularity run",
        "--cleanenv",
        "-B $SLURM_TMPDIR:/work",  # use SLURM_TMPDIR to overcome scratch file number limit
        "-B /etc/pki:/etc/pki/",
    ]
)

slurm_preamble = """#!/bin/bash
#SBATCH --account={slurm_account}
#SBATCH --job-name={jobname}.job
#SBATCH --output={bids_root}/.slurm/{jobname}.out
#SBATCH --error={bids_root}/.slurm/{jobname}.err
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-cpu={mem_per_cpu}G
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={email}


export SINGULARITYENV_TEMPLATEFLOW_HOME=/home/bidsapp/.cache/templateflow/
"""


def load_bidsignore(bids_root):
    """Load .bidsignore file from a BIDS dataset, returns list of regexps"""
    bids_ignore_path = bids_root / ".bidsignore"
    if bids_ignore_path.exists():
        import re
        import fnmatch

        bids_ignores = bids_ignore_path.read_text().splitlines()
        return tuple(
            [
                re.compile(fnmatch.translate(bi))
                for bi in bids_ignores
                if len(bi) and bi.strip()[0] != "#"
            ]
        )
    return tuple()

def write_mriqc_job(layout, subject, session, args, type='func'):
    study = os.path.basename(layout.root)
    job_specs = dict(
        slurm_account=args.slurm_account,
        jobname=f"mriqc_study-{study}_sub-{subject}_ses-{session}",
        email=args.email,
        bids_root=layout.root,
    )
    job_specs.update(MRIQC_REQ)
    job_path = os.path.join(layout.root, SLURM_JOB_DIR, f"{job_specs['jobname']}.sh")

    derivatives_path = os.path.join(layout.root, "derivatives", args.derivatives_name)

    pybids_cache_path = os.path.join(layout.root, PYBIDS_CACHE_PATH)

    mriqc_singularity_path = args.container or MRIQC_DEFAULT_SINGULARITY_PATH

    if type == 'anat':
        acqs = ['T1w']
    else:
        acqs = ['bold']

    with open(job_path, "w") as f:
        f.write(slurm_preamble.format(**job_specs))

        f.write(
            " ".join(
                [
                    SINGULARITY_CMD_BASE,
                    f"-B {layout.root}:{layout.root}",
                    mriqc_singularity_path,
                    "-w /work",
                    f"--participant-label {subject}",
                    f"--session-id {session}",
                    f"--omp-nthreads {job_specs['omp_nthreads']}",
                    f"--nprocs {job_specs['cpus']}",
                    f"-m {' '.join(acqs)}",
                    f"--mem_gb {job_specs['mem_per_cpu']*job_specs['cpus']}",
                    "--no-sub", # no internet on compute nodes
                    layout.root,
                    derivatives_path,
                    "participant",
                    "\n",
                ]
            )
        )
    return job_path



def submit_slurm_job(job_path):
    return subprocess.run(["sbatch", job_path])


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="submit mriqc jobs",
    )
    parser.add_argument(
        "bids_path", type=pathlib.Path, help="BIDS folder to run mriqc on."
    )
    parser.add_argument(
        "derivatives_name",
        type=pathlib.Path,
        help="name of the output folder in derivatives.",
    )
    parser.add_argument("preproc", help="anat or func")
    parser.add_argument(
        "--slurm-account",
        action="store",
        default="rrg-pbellec",
        help="SLURM account for job submission",
    )
    parser.add_argument("--email", action="store", help="email for SLURM notifications")
    parser.add_argument(
        "--container", action="store", help="mriqc singularity container"
    )
    parser.add_argument(
        "--participant-label",
        action="store",
        nargs="+",
        help="a space delimited list of participant identifiers or a single "
        "identifier (the sub- prefix can be removed)",
    )
    parser.add_argument(
        "--session-label",
        action="store",
        nargs="+",
        help="a space delimited list of session identifiers or a single "
        "identifier (the ses- prefix can be removed)",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force pyBIDS reset_database and reindexing",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Generate scripts, do not submit SLURM jobs, for testing.",
    )
    return parser.parse_args()


def run_mriqc(layout, args, pipe="anat"):

    subjects = args.participant_label
    if not subjects:
        subjects = layout.get_subjects()

    for subject in subjects:
        if args.session_label:
            sessions = args.session_label
        else:
            sessions = layout.get_sessions(subject=subject)



        for session in sessions:
            yield write_mriqc_job(layout, subject, session, args, type=pipe)

def main():

    args = parse_args()

    pybids_cache_path = os.path.join(args.bids_path, PYBIDS_CACHE_PATH)

    layout = bids.BIDSLayout(
        args.bids_path,
        database_path=pybids_cache_path,
        reset_database=args.force_reindex,
        ignore=(
            "code",
            "stimuli",
            "sourcedata",
            "models",
            re.compile(r"^\."),
        )
        + load_bidsignore(args.bids_path),
    )

    job_path = os.path.join(layout.root, SLURM_JOB_DIR)
    if not os.path.exists(job_path):
        os.mkdir(job_path)
        # add .slurm to .gitignore
        with open(os.path.join(layout.root, ".gitignore"), "a+") as f:
            f.seek(0)
            if not any([SLURM_JOB_DIR in l for l in f.readlines()]):
                f.write(f"{SLURM_JOB_DIR}\n")

    for job_file in run_mriqc(layout, args, args.preproc):
        if not args.no_submit:
            submit_slurm_job(job_file)


if __name__ == "__main__":
    main()
