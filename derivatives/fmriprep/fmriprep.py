import os
import sys
import argparse
import bids
import subprocess
import json

script_dir = os.path.dirname(__file__)

PYBIDS_CACHE_PATH = '.pybids_cache'
SLURM_JOB_DIR = '.slurm'

SMRIPREP_REQ = {'cpus': 4, 'mem_per_cpu': 4096, 'time':'8:00:00'}
FMRIPREP_REQ = {'cpus': 4, 'mem_per_cpu': 4096, 'time':'36:00:00'}

FMRIPREP_VERSION = "fmriprep-20.0.1-lts"
FMRIPREP_SINGULARITY_PATH = os.path.abspath(os.path.join(script_dir, f"../../containers/{FMRIPREP_VERSION}.simg"))
BIDS_FILTERS_FILE = os.path.join(script_dir, 'bids_filters.json')
TEMPLATEFLOW_HOME = os.path.join(
    os.environ.get(
        'SCRATCH',
        os.path.join(os.environ['HOME'],'.cache')),
    'templateflow')
OUTPUT_TEMPLATES = ['MNI152NLin2009cAsym', 'fsLR']
SINGULARITY_CMD_BASE = " ".join([
    "singularity run",
    "--cleanenv",
    f"-B {TEMPLATEFLOW_HOME}:/templateflow",
    f"-B /etc/pki:/etc/pki/",
    ])

slurm_preamble = """#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name={jobname}.job
#SBATCH --output={bids_root}/.slurm/{jobname}.out
#SBATCH --error={bids_root}/.slurm/{jobname}.err
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-cpu={mem_per_cpu}M
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={email}

export SINGULARITYENV_FS_LICENSE=$HOME/.freesurfer.txt
export SINGULARITYENV_TEMPLATEFLOW_HOME=/templateflow
"""

def write_anat_job(layout, subject, args):
    job_specs = dict(
        jobname = f"smriprep_sub-{subject}",
        email=args.email,
        bids_root=layout.root)
    job_specs.update(SMRIPREP_REQ)
    job_path = os.path.join(
        layout.root,
        SLURM_JOB_DIR,
        f"{job_specs['jobname']}.sh")

    derivatives_path = os.path.join("/data", 'derivatives', FMRIPREP_VERSION)


    # use json load/dump to copy filters (and validate json in the meantime)
    bids_filters_path = os.path.join(
        SLURM_JOB_DIR,
        "bids_filters.json")
    bids_filters = json.load(open(BIDS_FILTERS_FILE))
    with open(os.path.join(layout.root,bids_filters_path), 'w') as f:
        json.dump(bids_filters, f)

    with open(job_path, 'w') as f:
        f.write(slurm_preamble.format(**job_specs))
        f.write(" ".join([
            SINGULARITY_CMD_BASE,
            f"-B {layout.root}:/data",
            FMRIPREP_SINGULARITY_PATH,
            f"--participant-label {subject}",
            "--anat-only",
            f"--bids-filter-file {os.path.join('/data', bids_filters_path)}",
            "--cifti-output 91k",
            "--skip_bids_validation",
            f"--mem_mb {job_specs['mem_per_cpu']*job_specs['cpus']}",
            "/data",
            derivatives_path,
            "participant",
            ]))
    return job_path


def write_func_job(layout, subject, session, args):
    study = os.path.basename(layout.root)
    anat_path = os.path.join(
        os.path.dirname(layout.root),
        'anat',
        'derivatives',
        FMRIPREP_VERSION)
    derivatives_path = os.path.join(layout.root, 'derivatives', FMRIPREP_VERSION)

    bold_runs = layout.get(
        subject=subject,
        session=session,
        extension=['.nii', '.nii.gz'],
        suffix='bold')
    n_runs = len(bold_runs)
    run_lengths = [run.get_image().shape[-1] for run in bold_runs]

    job_specs = dict(
        jobname = f"fmriprep_study-{study}_sub-{subject}_ses-{session}",
        email = args.email,
        bids_root=layout.root)
    job_specs.update(FMRIPREP_REQ)

    job_path = os.path.join(
        layout.root,
        SLURM_JOB_DIR,
        f"{job_specs['jobname']}.sh")
    bids_filters_path = os.path.join(
        SLURM_JOB_DIR,
        f"{job_specs['jobname']}_bids_filters.json")

    # filter for session
    bids_filters = json.load(open(BIDS_FILTERS_FILE))
    bids_filters['bold'].update({'session': session})
    with open(os.path.join(layout.root, bids_filters_path), 'w') as f:
        json.dump(bids_filters, f)

    with open(job_path, 'w') as f:
        f.write(slurm_preamble.format(**job_specs))
        f.write(" ".join([
            SINGULARITY_CMD_BASE,
            f"-B {layout.root}:/data",
            f"-B {anat_path}:/anat",
            FMRIPREP_SINGULARITY_PATH,
            f"--participant-label {subject}",
            f"--anat-derivatives /anat",
            f"--bids-filter-file {os.path.join('/data', bids_filters_path)}",
            "--cifti-output 91k",
            "--notrack",
            "--skip_bids_validation",
            f"--mem_mb {job_specs['mem_per_cpu'] * job_specs['cpus']}",
            "/data",
            derivatives_path,
            "participant",
            ]))

    return job_path

def submit_slurm_job(job_path):
    return subprocess.run(["sbatch", job_path])

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='submit smriprep jobs')
    parser.add_argument('bids_path',
                   help='BIDS folder to run smriprep on.')
    parser.add_argument('preproc',
                   help='anat or func')
    parser.add_argument(
        '--email', action='store',
        help='email for SLURM notifications')
    parser.add_argument(
        '--container', action='store',
        help='fmriprep singularity container')
    parser.add_argument(
        '--participant-label', action='store', nargs='+',
        help='a space delimisted list of participant identifiers or a single '
             'identifier (the sub- prefix can be removed)')
    parser.add_argument(
        '--force-reindex', action='store_true',
        help='Force pyBIDS reset_database and reindexing')
    parser.add_argument(
        '--no-submit', action='store_true',
        help='Generate scripts, do not submit SLURM jobs, for testing.')
    return parser.parse_args()

def run_smriprep(layout, args):

    subjects = args.participant_label
    if not subjects:
        subjects = layout.get_subjects()

    for subject in subjects:
        #if TODO: check if derivative already exists for that subject
        job_path = write_anat_job(layout, subject, args)
        if not args.no_submit:
            submit_slurm_job(job_path)

def run_fmriprep(layout, args):

    subjects = args.participant_label
    if not subjects:
        subjects = layout.get_subjects()

    for subject in subjects:
        #if TODO: check if derivative already exists for that subject

        sessions = layout.get_sessions(subject=subject)
        for session in sessions:
            write_func_job(layout, subject, session, args)
            if not args.no_submit:
                submit_slurm_job(job_path)

def main():

    args = parse_args()

    pybids_cache_path = os.path.join(args.bids_path, PYBIDS_CACHE_PATH)

    layout = bids.BIDSLayout(
        args.bids_path,
        database_path=pybids_cache_path,
        reset_database=args.force_reindex,
        index_metadata=False,
        validate=False)

    job_path = os.path.join(
        layout.root,
        SLURM_JOB_DIR)
    if not os.path.exists(job_path):
        os.mkdir(job_path)
        # add .slurm to .gitignore
        with open(os.path.join(layout.root, '.gitignore'), 'a+') as f:
            f.seek(0)
            if not any([SLURM_JOB_DIR in l for l in f.readlines()]):
                f.write(f"{SLURM_JOB_DIR}\n")

    # prefectch templateflow templates
    os.environ['TEMPLATEFLOW_HOME'] = TEMPLATEFLOW_HOME
    import templateflow.api as tf_api
    tf_api.get(OUTPUT_TEMPLATES + ['tpl-OASIS30ANTs'])

    if args.preproc == 'anat':
        run_smriprep(layout, args)
    elif args.preproc =='func':
        run_fmriprep(layout, args)

if __name__ == "__main__":
    main()
