import os
import sys
import glob
import argparse
import bids
import subprocess
import json
import re
import pathlib
import datalad.api

script_dir = os.path.dirname(__file__)

PYBIDS_CACHE_PATH = ".pybids_cache"
SLURM_JOB_DIR = "code"

MRIQC_REQ = {"cpus": 8, "mem_per_cpu": 4, "time": "2:00:00", "omp_nthreads": 8}

MRIQC_DEFAULT_VERSION = "mriqc-24.0.2"

SINGULARITY_CMD_BASE = " ".join(
    [
        "singularity run",
        "--cleanenv",
        "-B $SLURM_TMPDIR:/work",  # use SLURM_TMPDIR to overcome scratch file number limit
        "-B /etc/pki:/etc/pki/",
    ]
)

SINGULARITY_CMD_BASE = " ".join(
    [
        "datalad containers-run "
        "-m 'mriqc_{subject_session}'",
        "-n bids-mriqc",
        "--input sourcedata/{study}/{subject_session}/fmap/",
        "--input sourcedata/{study}/{subject_session}/func/",
        "--output .",
        "--",
    ]
)

slurm_preamble = """#!/bin/bash
#SBATCH --account={slurm_account}
#SBATCH --job-name={jobname}.job
#SBATCH --output={derivatives_path}/code/{jobname}.out
#SBATCH --error={derivatives_path}/code/{jobname}.err
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-cpu={mem_per_cpu}G
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={email}


"""


datalad_pre = """
export LOCAL_DATASET=$SLURM_TMPDIR/${{SLURM_JOB_NAME//-/}}/
flock --verbose {ds_lockfile} datalad clone {output_repo} $LOCAL_DATASET
cd $LOCAL_DATASET
git-annex enableremote ria-beluga-storage
datalad get -s ria-beluga-storage -J 4 -n -r -R1 . # get sourcedata/* containers
if [ -d sourcedata/smriprep ] ; then
    datalad get -n sourcedata/smriprep sourcedata/smriprep/sourcedata/freesurfer
fi
git submodule foreach --recursive git annex dead here
git submodule foreach git annex enableremote ria-beluga-storage
git checkout -b $SLURM_JOB_NAME

"""

datalad_post = """
flock --verbose {ds_lockfile} datalad push -d ./ --to origin
if [ -d sourcedata/freesurfer ] ; then
    flock --verbose {ds_lockfile} datalad push -J 4 -d sourcedata/freesurfer $LOCAL_DATASET --to origin
fi 
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
    print(subject, session)
    study = os.path.basename(layout.root)
    job_specs = dict(
        slurm_account=args.slurm_account,
        jobname=f"mriqc_study-{study}_sub-{subject}_ses-{session}",
        email=args.email,
        bids_root=layout.root,
        study=study,
        subject=subject,
        subject_session=f"sub-{subject}" + (f"/ses-{session}" if session else ""),
        output_repo=args.output_repo,
        derivatives_path=args.output_path,
        ds_lockfile=os.path.join(args.output_repo.replace('ria+file://','').split('@')[0].replace('#~','/alias/'), '.datalad_lock'),
    )
    job_specs.update(MRIQC_REQ)
    job_path = os.path.join(args.output_path, SLURM_JOB_DIR, f"{job_specs['jobname']}.sh")


    pybids_cache_path = os.path.join(layout.root, PYBIDS_CACHE_PATH)

    if type == 'anat':
        acqs = ['T1w']
    else:
        acqs = ['bold']

    with open(job_path, "w") as f:
        f.write(slurm_preamble.format(**job_specs))
        f.write(datalad_pre.format(**job_specs))

        f.write(
            " ".join(
                [
                    SINGULARITY_CMD_BASE.format(**job_specs),
                    "-w workdir/",
                    f"--participant-label {subject}",
                    f"--session-id {session}",
                    f"--omp-nthreads {job_specs['omp_nthreads']}",
                    f"--nprocs {job_specs['cpus']}",
                    f"-m {' '.join(acqs)}",
                    f"--mem_gb {job_specs['mem_per_cpu']*job_specs['cpus']}",
                    "--no-datalad-get",
                    "--bids-filter-file code/bids_filters.json" if os.path.exists("code/bids_filters.json") else "",
                    "--no-sub", # no internet on compute nodes
                    str(args.bids_path.relative_to(args.output_path)),
                    './',
                    "participant",
                    "\n",
                ]
            )
        )
        f.write("mriqc_exitcode=$?\n")
        f.write(datalad_post.format(**job_specs))
        f.write("exit $mriqc_exitcode \n")

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
        "output_path",
        type=pathlib.Path,
        help="path of the output folder",
    )
    parser.add_argument(
        "--output-repo",
        help="path to the ria-store dataset.",
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

    job_path = os.path.join(args.output_path, SLURM_JOB_DIR)
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

    datalad.api.save(glob.glob('code/*mriqc*.sh'))
    datalad.api.push(to='ria-beluga')

if __name__ == "__main__":
    main()
