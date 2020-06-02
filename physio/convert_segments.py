# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""Utilities for biosignal data structure."""

import os
# from pandas import DataFrame.to_csv - .to_csv is an attribute of dataframe
from neurokit2 import read_acqknowledge
import h5py


def batch_parse(root, subject, ses=None, save_path=None):
    """
    Automated signal parsing for biopac recordings following BIDS format.

    Make sure the trigger channel is named "TTL" because it is hard-coded
    Parameters:
    ------------
    root : path
        main directory containing the biopac data (e.g. /home/user/dataset)
    subject : string
        name of path for a specific subject (e.g.'sub-03')
    ses : string
        name of acquisition session. Optional workflow for specific experiment
        default is None
    save_path: path
        root directory of
    """
    # Check directory
    if os.path.exists(root) is False:
        raise ValueError("Couldn't find the following directory: ",  root)
    # handle no save_path given
    if save_path is None:
        save_path = root
    # Check directory
    elif os.path.exists(save_path) is False:
        raise ValueError("Couldn't find the following directory: ", save_path)

    # List the files that have to be parsed
    files = list_sub(root, subject, ses)

    # Main loop iterating through files in each dict key returned by list_sub
    for exp in files:
        for file in exp:
            # reading acq, resampling at 1000Hz
            bio_df, fs = read_acqknowledge(os.path.join(
                                       root, subject, exp, file),
                                           sampling_rate=1000)  # resampling

            # initialize a df with TTL values over 1 (switch either ~0 or ~5)
            query_df = bio_df.query('TTL > 1')

            # Define session length - this list will be less
            # memory expensive to play with than dataframe
            session = query_df.index[0:-1]

            # a priori known minimal length of a single block
            block_len = fs * 180
            # maximal TR - the time distance between two adjacent TTL
            tr_period = fs * 2

            # Define session length and adjust with padding
            padding = fs * 9
            start = session[0]-padding
            end = session[-1]+padding

            parse_list = []

            # ascertain that session is longer than 3 min
            if len(session) > block_len:
                for time in range(len(session)-1):
                    time_delta = session[time+1] - session[time]

                    # if the time diff between two TTL values over 1
                    # is larger than TR, keep both indexes
                    if time_delta > tr_period:
                        parse_start = session[time]
                        parse_end = session[time+1]
                        # adjust the segmentation with padding
                        # parse start is end of run
                        parse_list += [(parse_start + padding,
                                        parse_end - padding)]

            # Parse  with the given indexes
            # Keep the first segment before scanner is turned on
            # the first block is always from start to first parse
            block1 = bio_df[start:parse_list[0][0]]

            # runs are runs in the session
            runs = []
            # push the resulting parsed dataframes in a list
            runs += [block1]
            for i in range(len(parse_list)):
                if i == len(parse_list):
                    runs += ([bio_df[parse_list[i][1]:end]])
                    break
                else:
                    runs += ([bio_df[parse_list[i][1]:parse_list[1+i][0]]])

            # changing channel names
            for idx, run in enumerate(runs):
                run.rename(columns={"PPG100C": 'PPG',
                                    "Custom, HLT100C - A 6": 'RSP',
                                    "GSR-EDA100C-MRI": 'EDA',
                                    "ECG100C": 'ECG'})

                # joining path and file name with readable Run index(01 to 0n)
                sep = '_'
                name = sep.join(subject, exp, f'Run{idx+1:02}')
                # saving the dataframe under specified dir and file name
                hf = h5py.File(os.path.join(save_path, subject, exp, name),
                               'w')  # write HDF5
                hf.create_dataset(name, data=run)
                hf.close()
                # notify user
                print('run', f'Run{idx+1:02}', 'in file ', file,
                      '\n in experiment:', exp, 'is parsed.',
                      '\n and saved at', save_path, '| sampling rate is :', fs)

    return files


def list_sub(root=None, sub=None, ses=None, type='.acq', show=False):
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

        # display the lists (optional)
        if show:
            for exp in ses_runs:
                print("list of files for session %s" % exp, ses_runs[exp])

        # return a dictionary of sessions each containing a list of files
        return ses_runs
