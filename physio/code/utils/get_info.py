# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""another util for neuromod phys data conversion."""

import sys
from list_sub import list_sub
import json
import logging
from CLI import _get_parser2
from pandas import read_csv
import os

LGR = logging.getLogger(__name__)


def get_info(root=None, sub=None, ses=None, show=True, save=None):
    """
    Get all volumes taken for a sub.

    get_info pushes the info necessary to execute the phys2bids multi-run
    workflow to a dictionary. It can save it to `_volumes_all-ses-runs.json`
    in a specified path, or be printed in your terminal.

    Arguments
    ---------
    root : str BIDS CODE
        root directory of dataset, like "home/user/dataset"
    sub : str BIDS CODE
        subject number, like "sub-01"
    ses : str BIDS CODE
        session name or number, like "ses-hcptrt1"
    show : bool
        if you want to print the dictionary
    save_path : path
        if you want to save the dictionary in json format

    Returns
    -------
    ses_runs_vols : dict
        number of processed runs, number of expected runs, number of trigger or
        volumes per run, sourcedata file location

    Example :
    >>> ses_runs_vols = get_info(root = "/home/user/dataset", sub = "sub-01")
    """
    # list matches for a whole subject's dir
    ses_runs_matches = list_sub(f"{root}/sourcedata/physio",
                                sub, ses, type='.tsv', show=True)

    # go to fmri matches and get entries for each run of a session
    nb_expected_runs = {}

    # iterate through sessions and get _matches.tsv with list_sub dict
    for exp in ses_runs_matches:
        df = read_csv(f"{root}/sourcedata/physio/{sub}/{exp}/"
                      f"{ses_runs_matches[exp][0]}", sep='\t')  # first item

        # initialize some info we need
        idx = 1
        nb_expected_volumes_run = {}
        # get _bold.json filename in the matches.tsv we just read
        for filename in df.iloc[:, 0]:
            # strip part of root path
            filename = str(filename).replace("/sourcedata/physio/", "")

            # troubleshoot unexisting paths
            if os.path.exists(f"{root}/{filename[:-7]}.json") is False:
                try:
                    if os.path.exists(f"{root}/{filename[:-11]}"
                                      "run-01_bold.json") is False:
                        with open(f"{root}/{filename[:-11]}"
                                  "run-02_bold.json") as f:
                            bold = json.load(f)
                    else:
                        with open(f"{root}/{filename[:-11]}"
                                  "run-01_bold.json") as f:
                            bold = json.load(f)
                # if there is no way to load the file, notify user
                # NOTE : THIS HAS TO BE LOGGED ; SAVE THE LOG!
                except:
                    print('skiping', root, filename[:-7])
                    continue
            # if everything went well we should be alright with this
            else:
                with open(f"{root}/{filename[:-7]}.json") as f:
                    bold = json.load(f)
            # we want to GET THE NB OF VOLUMES
            nb_expected_volumes_run[f'run-{idx:02}'
                                    ] = bold["time"
                                             ]["samples"
                                               ]["AcquisitionNumber"][-1]
            # not super elegant but does the trick - counts the nb of runs/ses
            idx += 1

        # push all info in run in dict
        nb_expected_runs[exp] = nb_expected_volumes_run
        nb_expected_runs[exp]['expect_runs'] = len(df)
        nb_expected_runs[exp]['processed_runs'] = idx-1
        # strip part of the name
        nb_expected_runs[exp]['in_file'] = str(df.iloc[[0], [1]]
                                               )[:str(df.iloc[[0], [1]]
                                                      ).find('\n0')].strip(" ")

    if show:
        print(nb_expected_runs)
    if save is not None:
        if os.path.exists(f"{save}{sub}") is False:
            os.mkdir(f"{save}{sub}")
        with open(f"{save}{sub}/{sub}_volumes_all-ses-runs.json", 'w') as fp:
            json.dump(json.dumps(nb_expected_runs, indent=4), fp)
    return nb_expected_runs


def _main(argv=None):
    options = _get_parser2().parse_args(argv)
    get_info(**vars(options))


if __name__ == '__main__':
    _main(sys.argv[1:])
