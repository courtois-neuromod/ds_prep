import bids
import argparse

PYBIDS_CACHE_PATH = '.pybids_cache'
PYDEFACE_SINGULARITY = 'pydeface-2.0.0.simg'

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    parser.add_argument('bids_path',
                   help='BIDS folder to convert.')
    parse.add_argument('--force-reindex', action='store_true',
                   help='Force pyBIDS reset_database and reindexing')
    return parser.parse_args()

def main():
    args = parse_args()

    pybids_cache_path = os.path.join(args.bids_path, PYBIDS_CACHE_PATH)

    layout = bids.BIDSLayout(
        args.bids_path
        database_path=pybids_cache_path,
        reset_database=args.force_reindex,
        index_metadata=False)

    all_anats = layout.get(datatype='anat', extension='nii.gz')
    for anat in all_anats:
        # mask or defaced file
        if anat.suffix=='defacemask' or anat.rec=='defaced':
            continue
        # defaced file already exists
        if len(layout.get(rec='defaced', **anat.entities)):
            continue
        cmd = 'pydeface --outfile %s -%s' anat.path


if __name__ == "__main__":
    main()
