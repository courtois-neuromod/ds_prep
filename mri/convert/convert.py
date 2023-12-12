import datalad.api
from heudiconv.main import workflow as heudiconv_workflow
from argparse import ArgumentParser
import pathlib
import tempfile
import multiprocessing
from functools import partial
from filelock import FileLock
import fileinput
from ..prepare.fill_intended_for import fill_intended_for, fill_b0_meta
import re

HEURISTICS_PATH = pathlib.Path(__file__).parent.resolve() / 'heuristics_unf.py'


def parse_args():
    docstr = ("""Example:
             convert.py ....""")
    parser = ArgumentParser(description=docstr)
    parser.add_argument(
        "--output-datalad",
        required=True,
        help="path to the dataset where to store the sessions",
    )

    parser.add_argument(
        "--ria-storage-remote",
        required=False,
        help="name of the ria storage remote",
    )
    
    parser.add_argument(
        '--files',
        nargs='+',
        required=True,
        type=pathlib.Path,
        help='Files (tarballs, dicoms) or directories containing files to '
             'process. Cannot be provided if using --dicom_dir_template.')

    parser.add_argument(
        '--nprocs',
        type=int,
        default=4,
        help='number of jobs to run in parallel with multiprocessing')

    parser.add_argument(
        "--b0-field-id",
        action="store_true",
        help="fill new BIDS B0FieldIdentifier instead of IntendedFor",
    )
    
    return parser.parse_args()


def single_session_job(input_file, output_datalad, ria_storage_remote, b0_field_id=False):
    session_name = input_file.stem.split('.')[0]
    remote_path = pathlib.Path(output_datalad.replace('ria+file://', '').replace('#~', '/alias/').split('@')[0])
    lock_path = remote_path / '.datalad_lock'
    file_lock = FileLock(lock_path)

    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            with file_lock:
                ds = datalad.api.install(path=tmpdirname, source=output_datalad)

            # enable the ria storage remote
            if ria_storage_remote:
                ds.repo.enable_remote(ria_storage_remote)
            # checkout a new branch
            ds.repo.checkout(session_name, options=['-b'])
            # this clone will be flush, better say it's already dead.
            ds.repo.set_remote_dead('here')

            heudiconv_params = dict(
                files=[str(input_file)],
                outdir=tmpdirname,
                bids_options=[],
                datalad=True,
                heuristic=str(HEURISTICS_PATH)
            )
            print(heudiconv_params)
            heudiconv_workflow(**heudiconv_params)

            fix_fmap_phase(ds)
            fix_complex_events(ds)
            if b0_field_id:
                fill_b0_meta(ds.pathobj)
                ds.save(message='fill B0Field* tags')
            else:
                fill_intended_for(ds.pathobj)
                ds.save(message='fill IntendedFor')
            fix_fmap_multiecho(ds, commit='HEAD~3')
                
            if ria_storage_remote:
                ds.push(to=ria_storage_remote, data='anything') #if deps is not properly set
            with file_lock:
                print('pushing')
                ds.push(to='origin', data='anything')
            ds.repo.call_annex(['unused'])
            ds.repo.call_annex(['dropunused', '--force', 'all'])
            ds.drop('./.heudiconv/', reckless='kill', recursive=True)
            ds.drop('.', recursive=True)
        print(f"processed {input_file}")
        return True
    except Exception as e:
        print(f"An error occur processing {input_file}")
        print(e)
        import traceback
        print(traceback.format_exc())
        return False

def fix_fmap_phase(ds):
    #### fix fmap phase data (sbref series will contain both and heudiconv auto name it)
    new_files = [(ds.pathobj / nf) for nf in ds.repo.call_git(['show','--name-only','HEAD','--format=oneline']).split('\n')[1:]]

    phase_glob = 'sub-*/ses-*/fmap/*_part-phase*'
    phase_files = [f for f in ds.pathobj.glob(phase_glob) if f in new_files]
    if not list(phase_files):
        return
    ds.repo.remove(phase_files)
    mag_fmaps = [f for f in ds.pathobj.glob('sub-*/ses-*/fmap/*_part-mag*') if f in new_files]
    for f in mag_fmaps:
        ds.repo.call_git(['mv', str(f), str(f).replace('_part-mag','')])

    scans_tsvs = [nf for nf in new_files if '_scans.tsv' in str(nf)]
    ds.unlock(scans_tsvs, on_failure='ignore')
    with fileinput.input(files=scans_tsvs, inplace=True) as f:
        for line in f:
            if not all([k in line for k in ['fmap/','_part-phase']]): # remove phase fmap
                if 'fmap/' in line:
                    line = line.replace('_part-mag', '')
                print(line, end='')
    ds.save(message='fix fmap phase')

def fix_fmap_multiecho(ds, commit=None):
    echo_glob = '**/fmap/*_echo-*'
    echo_files = ds.pathobj.glob(echo_glob)
    if commit:
        new_files = [(ds.pathobj / nf) for nf in ds.repo.call_git(['show','--name-only', commit,'--format=oneline']).split('\n')[1:]]
        echo_files = [f for f in echo_files if f in new_files]
    else:
        echo_files = list(echo_files)
    
    if not list(echo_files):
        return
    for f in echo_files:
        new_f = re.sub(r"^(.*/sub-.*(_ses-[^_]+))(_acq-([^_]*))(.*)(_echo-([0-9]))(_.*)$", r"\1_acq-\4Echo\7\5\8", str(f))
        ds.repo.call_git(['mv', str(f), new_f])

    if commit:
        scans_tsvs = [nf for nf in new_files if '_scans.tsv' in str(nf)]
    else:
        scans_tsvs = list(ds.pathobj.glob('**/*_scans.tsv'))
        
    ds.unlock(scans_tsvs, on_failure='ignore')
    with fileinput.input(files=scans_tsvs, inplace=True) as f:
        for line in f:
            if not all([k in line for k in ['fmap/','_part-phase']]): # remove phase fmap
                if 'fmap/' in line:
                    line = re.sub(r"^(.*/sub-.*(_ses-[^_]+))(_acq-([^_]*))(.*)(_echo-([0-9]))(_.*)$", r"\1_acq-\4Echo\7\5\8", line)
                print(line, end='')

    ds.save(message='fix fmap multiecho')    

def fix_complex_events(ds):
    # remove phase event files.
    new_files = [(ds.pathobj / nf) for nf in ds.repo.call_git(['show','--name-only','HEAD~1','--format=oneline']).split('\n')[1:]]
    phase_events = [f for f in ds.pathobj.glob('sub-*/ses-*/func/*_part-phase*_events.tsv') if f in new_files]
    if list(phase_events):
        ds.repo.remove(phase_events)
    # remove part-mag from remaining event files
    mag_events = [f for f in ds.pathobj.glob('sub-*/ses-*/func/*_part-mag*_events.tsv') if f in new_files]
    if len(mag_events):
        for f in mag_events:
            ds.repo.call_git(['mv', str(f), str(f).replace('_part-mag','')])
        ds.save(message='fix complex event files')
    
def main():

    args = parse_args()
    nprocs = args.nprocs

    pool = multiprocessing.Pool(nprocs)
    res = pool.map(
        partial(single_session_job,
                output_datalad=args.output_datalad,
                ria_storage_remote=args.ria_storage_remote,
                b0_field_id=args.b0_field_id),
        args.files)

    print("SUMMARY" + "#"*40)
    for r,f in zip(res, args.files):
        print(f"{f}: {'SUCCESS' if r else 'FAIL'}")

if __name__ == "__main__":
    main()

