import os
import sys
import glob
import argparse
import bids
import subprocess
import json
import re
import pathlib
import shutil
import datalad.api

script_dir = os.path.dirname(__file__)

PYBIDS_CACHE_PATH = ".pybids_cache"
SLURM_JOB_DIR = "code"

FREESURFER_REQ = {"cpus": 4, "mem_per_cpu": 4096, "time": "24:00:00", "omp_nthreads": 8}

SINGULARITY_CMD_BASE = " ".join(
    [
        "datalad containers-run "
        "-m 'fMRIPrep_{subject_session}'",
        "-n containers/bids-freesurfer",
    ] + [
        "--output .",
    ]
)

slurm_preamble = """#!/bin/bash
#SBATCH --account={slurm_account}
#SBATCH --job-name={jobname}.job
#SBATCH --output={derivatives_path}/code/{jobname}.out
#SBATCH --error={derivatives_path}/code/{jobname}.err
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-cpu={mem_per_cpu}M
#SBATCH --tmp=100G
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={email}

 
set -e -u -x

"""


datalad_pre = """
export LOCAL_DATASET=$SLURM_TMPDIR/${{SLURM_JOB_NAME//-/}}/
flock --verbose {ds_lockfile} datalad clone {output_repo} $LOCAL_DATASET
cd $LOCAL_DATASET
datalad get -s ria-beluga-storage -J 4 -n -r -R1 . # get sourcedata/* containers
git submodule foreach --recursive git annex dead here
git checkout -b $SLURM_JOB_NAME

git submodule foreach  --recursive git-annex enableremote ria-beluga-storage

"""

datalad_post = """
flock --verbose {ds_lockfile} datalad push -d ./ --to origin
"""
def load_bidsignore(bids_root, mode="python"):
    """Load .bidsignore file from a BIDS dataset, returns list of regexps"""
    bids_ignore_path = bids_root / ".bidsignore"
    if bids_ignore_path.exists():
        bids_ignores = bids_ignore_path.read_text().splitlines()
        if mode == 'python':
            import re
            import fnmatch
            
            return tuple(
                [
                    re.compile(fnmatch.translate(bi))
                    for bi in bids_ignores
                    if len(bi) and bi.strip()[0] != "#"
                ]
            )
        elif mode == 'bash':
            return [
                f"m/{bi}/"
                for bi in bids_ignores
                if len(bi) and bi.strip()[0] != "#"
            ]
    return tuple()


def write_freesurfer_job(layout, subject, session, args):
    derivatives_path = os.path.realpath(os.path.abspath(args.output_path))

    study = os.path.basename(layout.root)
    
    job_specs = dict(
        study=study,
        subject=subject,
        subject_session=f"sub-{subject}" + (f"/ses-{session}" if session else "/ses-*"),
        slurm_account=args.slurm_account,
        jobname=f"freesurfer_{args.step}_sub-{subject}"+(f"_ses-{session}" if session else ""),
        email=args.email,
        bids_root=layout.root,
        output_repo=args.output_repo,
        derivatives_path=derivatives_path,
        ds_lockfile=os.path.join(args.output_repo.replace('ria+file://','').replace('#~','/alias/').split('@')[0], '.datalad_lock'),
    )
    job_specs.update(FREESURFER_REQ)

    job_path = os.path.join(derivatives_path, SLURM_JOB_DIR, f"{job_specs['jobname']}.sh")


    pybids_cache_path = os.path.join(layout.root, PYBIDS_CACHE_PATH)

    with open(job_path, "w") as f:
        f.write(slurm_preamble.format(**job_specs))
        f.write(datalad_pre.format(**job_specs))

        
        f.write(
            " ".join(
                [
                    SINGULARITY_CMD_BASE.format(**job_specs),
                    # too large scope, but only a few MB unnecessary pulled
                    "--input 'sourcedata/{study}/{subject_session}/anat/*_T1w.nii.gz'".format(**job_specs),
                    "--input 'sourcedata/{study}/{subject_session}/anat/*_T2w.nii.gz'".format(**job_specs),
                    "--input 'sourcedata/{study}/{subject_session}/anat/*_FLAIR.nii.gz'".format(**job_specs),
                    "--",
                    str(args.bids_path.relative_to(args.output_path)),
                    "./",
                    "participant",
                    f"--steps {args.step}",
                    "--refine_pial",
                    "--reconstruction_label norm",
                    "--refine_pial_reconstruction_label norm",
                    "--hires_mode enable",
                    f"--participant_label {subject}",
                    f"--session_label {session}" if session else "",
                    "--skip_bids_validator",
                    "--license_file", 'code/freesurfer.license',
                    f"--n_cpus {job_specs['cpus']}",
                    "\n",
                ]
            )
        )
        f.write("freesurfer_exitcode=$?\n")
        f.write(datalad_post.format(**job_specs))

        return job_path


def submit_slurm_job(job_path):
    return subprocess.run(["sbatch", job_path])


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="submit freesurfer jobs",
    )
    parser.add_argument(
        "bids_path", type=pathlib.Path, help="BIDS folder to run smriprep on."
    )
    parser.add_argument(
        "output_path",
        type=pathlib.Path,
        help="name of the output folder in derivatives.",
    )

    parser.add_argument(
        "--output-repo",
        help="path to the ria-store dataset.",
    )

    parser.add_argument(
        "step",
        choices=['cross-sectional', 'template', 'longitudinal'],
        help="which step of longitudinal pipeline to run")
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


def run_freesurfer(layout, args, step="cross-sectional"):

    subjects = args.participant_label
    if not subjects:
        subjects = layout.get_subjects()

    for subject in subjects:
        if args.session_label:
            sessions = args.session_label
        else:
            sessions = layout.get_sessions(subject=subject)
        if len(sessions) == 0:
            sessions = [None]

        if step == "cross-sectional":
            for session in sessions:
                # if TODO: check if derivative already exists for that subject
                job_path = write_freesurfer_job(layout, subject, session, args)
                yield job_path
        else:
            job_path = write_freesurfer_job(layout, subject, None, args)
            yield job_path


def main():

    args = parse_args()

    pybids_cache_path = os.path.join(args.bids_path, PYBIDS_CACHE_PATH)

    layout = bids.BIDSLayout(
        args.bids_path,
        database_path=pybids_cache_path,
        reset_database=args.force_reindex,
        validate=False,
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
        # add code to .gitignore
        with open(os.path.join(layout.root, ".gitignore"), "a+") as f:
            f.seek(0)
            if not any([SLURM_JOB_DIR in l for l in f.readlines()]):
                f.write(f"{SLURM_JOB_DIR}\n")
    license_path = os.path.join(job_path, 'freesurfer.license')
    if not os.path.exists(license_path):
        shutil.copyfile(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../fmriprep', 'freesurfer.license'), license_path)

    for job_file in run_freesurfer(layout, args, args.step):
        if not args.no_submit:
            submit_slurm_job(job_file)

    datalad.api.save(glob.glob('code/*mriprep*.sh')+glob.glob('code/*bids_filters.json'))
    datalad.api.push(to='ria-beluga')

if __name__ == "__main__":
    main()
