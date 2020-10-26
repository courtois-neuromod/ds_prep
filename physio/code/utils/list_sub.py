# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""CLI for physio utils."""


import os
import logging
from CLI import _get_parser
import sys
import json

LGR = logging.getLogger(__name__)


def list_sub(root=None, sub=None, ses=None, type='.acq',
             show=False, save=False):
    """
    List a subject's files.

    Returns a dictionary entry for each session in a subject's directory
    Each entry is a list of files for a given subject/ses directory
    if ses is given, only one dictionary entry is returned

    Arguments
    ---------
    root :
        root directory of dataset, like "home/user/dataset"
    sub :
        subject number, like "sub-01"
    ses :
        session name or number, like "ses-hcptrt1"
    type :
        what file are we looking for. Default is biosignals from biopac
    show :
        if you want to print the dictionary
    save :
        if you want to save the dictionary in json format

    Returns
    -------
    ses_list :
        list of sessions in the subject's folder
    files_list :
        list of files by their name

    Example :
    >>> ses_runs = list_sub(root = "/home/user/dataset", sub = "sub-01")
    """
    # Check the subject's
    if os.path.exists(os.path.join(root, sub)) is False:
        raise ValueError("Couldn't find the subject's path \n",
                         os.path.join(root, sub))
    file_list = []
    ses_runs = {}
    ses_list = os.listdir(os.path.join(root, sub))

    # list files in only one session
    if ses is not None:
        dir = os.path.join(root, sub, ses)

        # if the path exists, list .acq files
        if os.path.exists(dir):
            for filename in os.listdir(dir):

                if filename.endswith(type):
                    file_list += [filename]
            if show:
                print("list of sessions in subjet's directory: ", ses_list)
                print('list of files in the session:', file_list)

            # return a dictionary entry for the specified session
            files = {str(ses): file_list}
            return files
        else:
            print("list of sessions in subjet's directory: ", ses_list)
            raise Exception("Session path you gave does not exist")

    # list files in all sessions (or here, exp for experiments)
    else:
        for exp in ses_list:
            # re-initialize the list
            file_list = []
            # iterate through directory's content
            for filename in os.listdir(os.path.join(root, sub, exp)):
                if filename.endswith(type):
                    file_list += [filename]

            # save the file_list as dict item
            ses_runs[exp] = file_list

        # sort dict entries numerically with ses ### code
        ses_runs = {key: value for key, value in
                    sorted(ses_runs.items(),
                           key=lambda item: int(item[0][-3:]))}
        # display the lists (optional)
        if show:
            for exp in ses_runs:
                print("list of files for session %s" % exp, ses_runs[exp])

        # Save the dict under temporary folder at sourcedata
        # ERRATUM : change sourcedata for folder where i have write access
        if save:
            filename = os.path.join(root, 'tmp', f'{sub}_info.json'
            if os.path.exists(os.path.join(root, 'tmp')) is False:
                os.mkdir(os.path.join(root, 'tmp'))

            with open(filename,'w', encoding='utf-8') as f:
                json.dump(ses_runs, f, indent=4)

        # return a dictionary of sessions each containing a list of files
        return ses_runs

#  Get arguments


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    list_sub(**vars(options))


if __name__ == '__main__':
    _main(sys.argv[1:])
