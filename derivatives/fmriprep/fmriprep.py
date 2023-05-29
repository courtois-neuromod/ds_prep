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

SMRIPREP_REQ = {"cpus": 8, "mem_per_cpu": 4096, "time": "24:00:00", "omp_nthreads": 8}
FMRIPREP_REQ = {"cpus": 12, "mem_per_cpu": 4096, "time": "12:00:00", "omp_nthreads": 8}

BIDS_FILTERS_FILE = os.path.join(script_dir, "bids_filters.json")

TEMPLATEFLOW_HOME = os.path.join(
    os.environ.get("SCRATCH", os.path.join(os.environ["HOME"], ".cache")),
    "templateflow",
)
OUTPUT_TEMPLATES = ["MNI152NLin2009cAsym", "T1w:res-iso2mm"]
REQUIRED_TEMPLATES = ["MNI152NLin2009cAsym", "OASIS30ANTs", "fsLR", "fsaverage", "MNI152NLin6Asym"]
SINGULARITY_CMD_BASE = " ".join(
    [
        "datalad containers-run "
        "-m 'fMRIPrep_{subject_session}'",
        "-n containers/bids-fmriprep",
    ] + [
        "--input sourcedata/templateflow/tpl-%s/"% tpl for tpl in REQUIRED_TEMPLATES
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
export SINGULARITYENV_TEMPLATEFLOW_HOME="${{LOCAL_DATASET}}/sourcedata/templateflow/"
flock --verbose {ds_lockfile} datalad clone {output_repo} $LOCAL_DATASET
cd $LOCAL_DATASET
datalad get -s ria-beluga-storage -J 4 -n -r -R1 . # get sourcedata/* containers
datalad get -s ria-beluga-storage -J 4 -r sourcedata/templateflow/tpl-{{{templates}}}
if [ -d sourcedata/smriprep ] ; then
    datalad get -n sourcedata/smriprep sourcedata/smriprep/sourcedata/freesurfer
fi
git submodule foreach --recursive git annex dead here
git checkout -b $SLURM_JOB_NAME
if [ -d sourcedata/freesurfer ] ; then
  git -C sourcedata/freesurfer checkout -b $SLURM_JOB_NAME
fi

git submodule foreach  --recursive git-annex enableremote ria-beluga-storage

"""

datalad_post = """
flock --verbose {ds_lockfile} datalad push -d ./ --to origin
if [ -d sourcedata/freesurfer ] ; then
    flock --verbose {ds_lockfile} datalad push -J 4 -d sourcedata/freesurfer --to origin
fi 
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


def write_job_footer(fd, jobname):
    # TODO: copy resource monitor output
    fd.write(
        f"if [ -e $LOCAL_DATASET/workdir/fmriprep_wf/resource_monitor.json ] ; then cp $LOCAL_DATASET/workdir/fmriprep_wf/resource_monitor.json /scratch/{os.environ['USER']}/{jobname}_resource_monitor.json ; fi \n"
    )
    fd.write(
        f"if [ $fmriprep_exitcode -ne 0 ] ; then cp -R $LOCAL_DATASET /scratch/{os.environ['USER']}/{jobname} ; fi \n"
    )
    fd.write("exit $fmriprep_exitcode \n")


def write_fmriprep_job(layout, subject, args, anat_only=True, longitudinal=False):
    derivatives_path = os.path.realpath(os.path.abspath(args.output_path))

    study = os.path.basename(layout.root)
    
    job_specs = dict(
        study=study,
        subject=subject,
        subject_session=f"sub-{subject}" + (f"/ses-{args.session_label}" if args.session_label else "/ses-*"),
        slurm_account=args.slurm_account,
        jobname=f"{'s' if anat_only else 'f'}mriprep_sub-{subject}",
        email=args.email,
        bids_root=layout.root,
        output_repo=args.output_repo,
        derivatives_path=derivatives_path,
        ds_lockfile=os.path.join(args.output_repo.replace('ria+file://','').replace('#~','/alias/').split('@')[0], '.datalad_lock'),
        TEMPLATEFLOW_HOME=TEMPLATEFLOW_HOME,
        templates=",".join(REQUIRED_TEMPLATES),
    )
    job_specs.update(SMRIPREP_REQ)
    if args.longitudinal:
        job_specs['time'] = '72:0:0'

    job_path = os.path.join(derivatives_path, SLURM_JOB_DIR, f"{job_specs['jobname']}.sh")

    # use json load/dump to copy filters (and validate json in the meantime)
    bids_filters_path = os.path.join(
        SLURM_JOB_DIR,
        "bids_filters.json")
    bids_filters = json.load(open(BIDS_FILTERS_FILE))
    with open(bids_filters_path, 'w') as f:
        json.dump(bids_filters, f)

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
                    "-w ./workdir",
                    f"--participant-label {subject}",
                    "--anat-only" if anat_only else "",
#                    f"--bids-database-dir {pybids_cache_path}",
                    f"--bids-filter-file {bids_filters_path}",
                    "--output-layout bids",
                    "--output-spaces",
                    " ".join(OUTPUT_TEMPLATES), "MNI152NLin6Asym",
                    "--cifti-output 91k",
                    "--skip_bids_validation",
                    "--write-graph",
                    f"--omp-nthreads {job_specs['omp_nthreads']}",
                    f"--nprocs {job_specs['cpus']}",
                    f"--mem_mb {job_specs['mem_per_cpu']*job_specs['cpus']}",
                    "--fs-license-file", 'code/freesurfer.license',
                    "--longitudinal" if longitudinal else "",
                    str(args.bids_path.relative_to(args.output_path)),
                    "./",
                    "participant",
                    "\n",
                ]
            )
        )
        f.write("fmriprep_exitcode=$?\n")
        f.write(datalad_post.format(**job_specs))
        write_job_footer(f, job_specs["jobname"])

        return job_path


def write_func_job(layout, subject, session, args):
    outputs_exist = False
    study = os.path.basename(layout.root)

    derivatives_path = os.path.realpath(args.output_path)
    anat_path = os.path.join(
        derivatives_path,
        "sourcedata",
        "smriprep",
    )

    bold_runs = layout.get(
        subject=subject,
        session=session if session else bids.layout.Query.NONE,
        extension=[".nii", ".nii.gz"],
        suffix="bold",
    )
    contains_phase_data = False
    if any([b.entities.get('part')=='phase' for b in bold_runs]):
        contains_phase_data = True
        print("Dataset contains phase bold(s), selecting only part-mag bolds.")
        bold_runs = layout.get(
            subject=subject,
            session=session if session else bids.layout.Query.NONE,
            extension=[".nii", ".nii.gz"],
            suffix="bold",
            part='mag',
        )
    if len(bold_runs) == 0:
        print(f"No bold runs found for {subject} {session}")

    bold_derivatives = []
    for bold_run in bold_runs:
        entities = bold_run.entities
        entities = [
            (ent, entities[ent])
            for ent in ["subject", "session", "task", "run"]
            if ent in entities
        ]
        preproc_entities = entities + [
            ("space", OUTPUT_TEMPLATES[0]),
            ("desc", "preproc"),
        ]
        dtseries_entities = entities + [("space", "fsLR"), ("den", "91k")]
        func_path = os.path.join(
            derivatives_path,
            "fmriprep",
            f"sub-{subject}",
            f"ses-{session}",
            "func",
        )

        
        preproc_path = os.path.join(
            func_path,
            "_".join(
                [
                    "%s-%s" % (k[:3] if k in ["subject", "session"] else k, v)
                    for k, v in preproc_entities
                ]
            )
            + "_bold.nii.gz",
        )
        dtseries_path = os.path.join(
            func_path,
            "_".join(
                [
                    "%s-%s" % (k[:3] if k in ["subject", "session"] else k, v)
                    for k, v in dtseries_entities
                ]
            )
            + "_bold.dtseries.nii",
        )
        # test if file or symlink (even broken if git-annex and not pulled)
        bold_deriv = os.path.lexists(preproc_path) and os.path.lexists(dtseries_path)
        if bold_deriv:
            print(
                f"found existing derivatives for {bold_run.path} : {preproc_path}, {dtseries_path}"
            )
        bold_derivatives.append(bold_deriv)
    outputs_exist = all(bold_derivatives)
    # n_runs = len(bold_runs)
    # run_shapes = [run.get_image().shape for run in bold_runs]
    # run_lengths = [rs[-1] for rs in run_shapes]

    subject_session = f"sub-{subject}" + (f"/ses-{session}" if session not in [None,'*'] else "")
    job_specs = dict(
        study=study,
        subject=subject,
        session=session,
        subject_session=subject_session,
        slurm_account=args.slurm_account,
        jobname=f"fmriprep_study-{study}_sub-{subject}"+ (f"_ses-{session}" if session not in [None, '*'] else ""),
        email=args.email,
        bids_root=layout.root,
        derivatives_path=derivatives_path,
        output_repo=args.output_repo,
        ds_lockfile=os.path.join(args.output_repo.replace('ria+file://','').replace('#~','/alias/').split('@')[0], '.datalad_lock'),
        TEMPLATEFLOW_HOME=TEMPLATEFLOW_HOME,
        templates=",".join(REQUIRED_TEMPLATES),
    )
    job_specs.update(FMRIPREP_REQ)

    job_path = os.path.join(derivatives_path, SLURM_JOB_DIR, f"{job_specs['jobname']}.sh")
    bids_filters_path = os.path.join(
        SLURM_JOB_DIR,
        f"{job_specs['jobname']}_bids_filters.json"
    )

    pybids_cache_path = os.path.join(layout.root, PYBIDS_CACHE_PATH)

    # filter for session
    bids_filters = json.load(open(BIDS_FILTERS_FILE))
    for acq in ["bold","sbref","fmap"]:
        bids_filters[acq].update({"session": [session]})
    if contains_phase_data:
        for acq in ["bold","sbref"]:
            bids_filters[acq].update({"part": "mag"})
    with open(bids_filters_path, "w") as f:
        json.dump(bids_filters, f)


    with open(job_path, "w") as f:
        f.write(slurm_preamble.format(**job_specs))
        f.write(datalad_pre.format(**job_specs))

        f.write(
            " ".join(
                [
                    SINGULARITY_CMD_BASE.format(**job_specs),
                    f"--input 'sourcedata/{study}/{subject_session}/fmap/'",
                    f"--input 'sourcedata/{study}/{subject_session}/func/'",
                    
                    f"--input 'sourcedata/smriprep/sub-{subject}/anat/'",
                    f"--input sourcedata/smriprep/sourcedata/freesurfer/fsaverage/",
                    f"--input sourcedata/smriprep/sourcedata/freesurfer/sub-{subject}/",
                    "--",
                    "-w ./workdir",
                    f"--participant-label {subject}",
                    "--anat-derivatives ./sourcedata/smriprep",
                    "--fs-subjects-dir ./sourcedata/smriprep/sourcedata/freesurfer",
#                    f"--bids-database-dir {pybids_cache_path}",
                    f"--bids-filter-file {bids_filters_path}",
                    "--output-layout bids",
                    "--ignore slicetiming" if not args.slicetiming else "",
                    "--use-syn-sdc",
                    "--output-spaces",
                    *OUTPUT_TEMPLATES,
                    "--cifti-output 91k",
                    "--notrack",
                    "--write-graph",
                    "--skip_bids_validation",
                    f"--omp-nthreads {job_specs['omp_nthreads']}",
                    f"--nprocs {job_specs['cpus']}",
                    f"--mem_mb {job_specs['mem_per_cpu']*job_specs['cpus']}",
                    "--fs-license-file", 'code/freesurfer.license',
                    # monitor resources to design a heuristic for runtime/cpu/ram of func data
                    #"--resource-monitor",
                    str(args.bids_path.relative_to(args.output_path)),
                    "./",
                    "participant",
                    "\n",
                ]
            )
        )
        f.write("fmriprep_exitcode=$?\n")
        f.write(datalad_post.format(**job_specs))
        write_job_footer(f, job_specs["jobname"])

        
    return job_path, outputs_exist


def submit_slurm_job(job_path):
    return subprocess.run(["sbatch", job_path])


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="submit smriprep jobs",
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

    parser.add_argument("preproc", help="anat or func")
    parser.add_argument(
        "--slurm-account",
        action="store",
        default="rrg-pbellec",
        help="SLURM account for job submission",
    )
    parser.add_argument("--email", action="store", help="email for SLURM notifications")
    parser.add_argument(
        "--container", action="store", help="fmriprep singularity container"
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
        "--slicetiming",
        action="store_true",
        help="Activate slicetiming correction",
    )
    parser.add_argument(
        "--longitudinal",
        action="store_true",
        help="Use smriprep longitudinal pipeline",
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


def run_fmriprep(layout, args, pipe="all"):

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

        if pipe == "func":
            for session in sessions:
                # if TODO: check if derivative already exists for that subject
                job_path, outputs_exist = write_func_job(layout, subject, session, args)
                if outputs_exist:
                    print(
                        f"all output already exists for sub-{subject} ses-{session}, not rerunning"
                    )
                    continue
                yield job_path
        elif pipe == "anat":
            yield write_fmriprep_job(layout, subject, args, anat_only=True, longitudinal=args.longitudinal)
        elif pipe == "all":
            yield write_fmriprep_job(layout, subject, args, anat_only=False, longitudinal=args.longitudinal)


def main():

    args = parse_args()

    pybids_cache_path = os.path.join(args.bids_path, PYBIDS_CACHE_PATH)

    """
    if not os.path.exists(pybids_cache_path):
        os.mkdir(pybids_cache_path)
        
    from datalad_container.containers_run import ContainersRun
    bids_root_container = pathlib.Path('/data').joinpath(args.bids_path.relative_to(args.output_path))
    
    pybids_cache_path_container = bids_root_container.joinpath(PYBIDS_CACHE_PATH)
    pybids_cache_cmd = [
        f"pybids layout",
        "--reset-db" if args.force_reindex else None ] + \
        [f"--ignore {p}" for p in ("code","stimuli","sourcedata","models","m/^\./") + load_bidsignore(args.bids_path, mode='bash')] + \
        [
            str(bids_root_container),
            str(pybids_cache_path_container),
        ]
    print(" ".join(filter(bool, pybids_cache_cmd)))
    ContainersRun()(
        " ".join(filter(bool, pybids_cache_cmd)),
        container_name='containers/fmriprep-lts',
        )
    """

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
        shutil.copyfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'freesurfer.license'), license_path)

    for job_file in run_fmriprep(layout, args, args.preproc):
        if not args.no_submit:
            submit_slurm_job(job_file)

    datalad.api.save(glob.glob('code/*mriprep*.sh')+glob.glob('code/*bids_filters.json'))
    datalad.api.push(to='ria-beluga')

if __name__ == "__main__":
    main()
