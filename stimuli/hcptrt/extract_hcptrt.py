#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Convert txt data from eprime to tsv using convert_eprime.
    git@github.com:tsalo/convert-eprime.git

    HCPTRT tasks
    https://github.com/hbp-brain-charting/public_protocols

"""

import argparse
import json
import logging
import numpy as np
import os

from convert_eprime.convert import _text_to_df
import pandas as pd

column = "column"
value = "value"
formula = "formula"
when_no_value = "when_no_value"
type = "type"


def _build_args_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    p.add_argument("in_file", help="Task output (.txt) to convert.")
    p.add_argument(
        "in_task", help="Config JSON file defining the task you want to " "convert."
    )
    p.add_argument("out_file", help="output tsv file.")
    p.add_argument(
        "--extract_eprime",
        action="store_true",
        help="Extract eprime file into raw tsv file. \n"
        "It helps to create a config file.",
    )
    p.add_argument(
        "-f",
        dest="overwrite",
        action="store_true",
        help="Force overwriting of the output files.",
    )
    p.add_argument(
        "-v",
        action="store_true",
        dest="verbose",
        help="If set, produces verbose output.",
    )
    return p


def assert_task_exists(in_task):
    """
    Assert if in_task exist and return dict (config file).

    Parameters
    -------
    in_task : str
        Path to json config file.

    Returns
    -------
    task: Dict
        Dictionnary of
    """

    in_task_filename, in_task_extension = os.path.splitext(os.path.basename(in_task))

    if in_task_extension == ".json":
        fullpath = in_task
    else:
        raise IOError("Task has to be a json file or choose one from config" " folder.")

    if os.path.exists(fullpath):
        with open(fullpath, "r") as json_file:
            task = json.load(json_file)
    else:
        raise IOError("{} does not exist".format(in_task))

    return task


def assert_task_df(df_columns, json_dict):
    """
    Assert if json_dict is using column name that exists in df_columns.
    Does not look into ioi dict.

    Parameters
    ----------
    df_columns : int
        length of the pandas dataframe.

    json_dict : dict
        Dict to select rows of interest.

    Returns
    -------
    """

    allColumns = []

    for curr_event in json_dict["events"]:
        for curr_key in curr_event:
            if isinstance(curr_event[curr_key], list):
                for element in curr_event[curr_key]:
                    if column in element:
                        if isinstance(element[column], list):
                            allColumns = allColumns + element[column]
                        else:
                            allColumns.append(element[column])
            else:
                if column in curr_event[curr_key]:
                    if isinstance(curr_event[curr_key][column], list):
                        allColumns = allColumns + curr_event[curr_key][column]
                    else:
                        allColumns.append(curr_event[curr_key][column])

    allColumns = list(dict.fromkeys(allColumns))

    for curr_column in allColumns:
        if not set([curr_column]).issubset(df_columns):
            raise IOError('Column: "{}" does not exist ' "into df".format(curr_column))


def get_ioi(df, ioi_dict):
    """
    Get Index of Interest from event dict.

    Parameters
    ----------
    df_len : int
        length of the pandas dataframe.

    ioi_dicts : list of dict
        Dict to select rows of interest.

    Returns
    -------
    ioi: pandas.core.series.Series
        Series of Index of interest.
    """
    indexes = np.zeros((len(df),), dtype=bool)

    columnName = ioi_dict[column]
    if isinstance(columnName, list):  # Column(list)
        if formula in ioi_dict:  # If formula
            if ioi_dict[formula] == "intersection":
                indexes = intersection_columns(df, ioi_dict)
            else:
                logging.error(
                    "function: get_ioi - If you have multiple "
                    "columns you need a formula to know what "
                    "should be done. Default merge"
                )
        else:
            ioi_serie = merge_columns(df, ioi_dict)
            indexes = ~np.isnan(ioi_serie.astype(np.float))

    elif isinstance(columnName, str):  # Column(str)
        if value in ioi_dict:  # Column(str) - Value
            indexes = np.zeros((len(df),))
            if isinstance(ioi_dict[value], str):  # Value(str)
                columnValues = [ioi_dict[value]]
            elif isinstance(ioi_dict[value], list):  # Value(list)
                columnValues = ioi_dict[value]

            for columnValue in columnValues:
                curr_index = df[columnName] == columnValue
                indexes = indexes | curr_index
        else:  # Column(str) - No Value
            indexes = df[columnName].astype(str) != "nan"

    return indexes


def get_durations(df, onset, duration_dict):
    """
    Get durations from event dict.

    Parameters
    ----------
    df : pandas DataFrame
        current DataFrame

    duration_dict : dict
        Dict to select rows or compute duration.

    Returns
    -------
    duration_serie: pandas.core.series.Series
        Series of Index with durations.
    """

    if isinstance(duration_dict[column], list):
        duration_serie = merge_columns(df, duration_dict)
    elif isinstance(duration_dict[column], str):
        duration_serie = df[duration_dict[column]]
        if formula in duration_dict:
            curr_formula = duration_dict[formula]
            if len(curr_formula) == 3:
                duration_serie = duration_serie.shift(periods=curr_formula[0])

            if curr_formula[1] == "subtract":
                duration_serie = duration_serie.astype(np.float) - onset.astype(
                    np.float
                )
            elif curr_formula[1] == "add":
                duration_serie = duration_serie.astype(np.float) + onset.astype(
                    np.float
                )

    if when_no_value in duration_dict:
        if duration_dict[when_no_value] == "median":
            duration_serie[np.isnan(duration_serie.astype(np.float))] = np.nanmedian(
                duration_serie.astype(np.float)
            )
        elif duration_dict[when_no_value] == "mean":
            duration_serie[np.isnan(duration_serie.astype(np.float))] = np.nanmean(
                duration_serie.astype(np.float)
            )

    duration_serie = duration_serie.rename("duration")
    return duration_serie


def intersection_columns(df, curr_dict):
    """
    Intersection of columns/values -> Get Index of Interest from event['ioi'].

    Parameters
    ----------
    df : pandas DataFrame
        Current DataFrame.

    curr_dict : dict
        Dictionnary with column names and values.

    Returns
    -------
    new_serie: pandas.core.series.Series
        Indexes of intersection of columns/values.
    """
    listColumns = curr_dict[column]
    listValues = curr_dict[value]

    new_serie = df[listColumns[0]] == listValues[0]
    listColumns = listColumns[1::]
    listValues = listValues[1::]
    for idx, curr_column in enumerate(listColumns):
        new_serie2 = df[curr_column] == listValues[idx]
        new_serie = new_serie & new_serie2

    return new_serie


def merge_columns(df, curr_dict):
    """
    Merge columns. Check if columns have the values with same indexes.

    Parameters
    ----------
    df : pandas DataFrame
        Current DataFrame.

    curr_dict : dict
        Dictionnary with column names and values.

    Returns
    -------
    new_serie: pandas.core.series.Series
        Merge serie.
    """
    curr_columns = curr_dict[column]
    if not isinstance(curr_columns, list):
        raise IOError("If you want to merge columns you need at least " "2 columns.")

    if value in curr_dict:  # Value exist
        curr_values = curr_dict[value]
        if isinstance(curr_values, list):  # Value (list)
            if len(curr_values) != len(curr_columns):
                raise IOError("List of values should match columns")
        elif isinstance(curr_values, str):  # Value (str)
            curr_values = [curr_values] * len(curr_columns)

        listColumns = curr_columns[1::]
        listValues = curr_values[1::]
        index_serie = df[curr_columns[0]] == curr_values[0]
        new_serie = df[curr_columns[0]]
        new_serie[new_serie != curr_values[0]] = "nan"

        for idx, curr_column in enumerate(listColumns):
            new_index = df[curr_column == listValues[idx]]
            new_serie = df[curr_column]
            new_serie[new_serie != listValues[idx]] = "nan"
            if np.any(new_index & index_serie):
                raise IOError("Columns contain values at the same index")
            else:
                index_serie = index_serie | new_index
                new_serie = new_serie.combine_first(new_serie)

    else:  # Value does not exist
        listColumns = curr_dict[column][1::]
        new_serie = df[curr_dict[column][0]]
        index_wo_nan = new_serie.astype(str) != "nan"

        for curr_column in listColumns:
            new_index = df[curr_column].astype(str) != "nan"
            if np.any(new_index & index_wo_nan):
                raise IOError("Columns contain values at the same index")
            else:
                index_wo_nan = index_wo_nan | new_index
                new_serie = new_serie.combine_first(df[curr_column])

    return new_serie


def get_key(df, columnName, key_dict):
    """
    Get Key from event dict.

    Parameters
    ----------
    df : pandas DataFrame
        Current DataFrame.

    columnName: str
        Rename column with this name.

    key_dict : dict
        Dict to select rows or compute duration.

    Returns
    -------
    key_serie: pandas.core.DataFrame
        Series from specific key.
    """

    if isinstance(key_dict[column], str):
        key_serie = df[key_dict[column]]
    elif isinstance(key_dict[column], list):
        key_serie = merge_columns(df, key_dict)
    else:
        logging.error("not coded yet - get_key")

    if type in key_dict:
        if key_dict[type] == "stim":
            key_serie = "../../stimulis/" + key_serie

        if key_dict[type] == "bloc":
            key_serie = key_serie.astype(np.float) - np.floor(
                key_serie.astype(np.float) / 2
            )

    if when_no_value in key_serie:
        if key_serie[when_no_value] == "median":
            key_serie[np.isnan(key_serie.astype(np.float))] = np.nanmedian(
                key_serie.astype(np.float)
            )
        elif key_serie[when_no_value] == "mean":
            key_serie[np.isnan(key_serie.astype(np.float))] = np.nanmean(
                key_serie.astype(np.float)
            )

    key_serie = key_serie.rename(columnName)

    return key_serie


def extract_event(df, curr_event, new_df, ttl):

    #  Get usefull indexes: ioi -> index of interest
    ioi = get_ioi(df, curr_event["ioi"])

    # Get onsets and durations
    onsets = get_key(df, "onset", curr_event["onset"])
    durations = get_durations(df, onsets, curr_event["duration"])

    # Creation of the event dataframe
    listOfType = [curr_event["name"]] * len(ioi)
    curr_event_df = pd.DataFrame(np.asarray(listOfType), columns=["type"])

    # Delete already used keys
    del curr_event["onset"]
    del curr_event["duration"]
    del curr_event["name"]
    del curr_event["ioi"]

    # Loop over all keys
    for curr_key in curr_event.keys():
        curr_event_df = curr_event_df.join(get_key(df, curr_key, curr_event[curr_key]))

    # Reduce all df to index of interest
    curr_event_df = curr_event_df[ioi]
    onsets = onsets[ioi]
    durations = durations[ioi]

    # Cast into np.float and divide by 1000
    onsets = onsets.astype(np.float).apply(_subTTL, var=ttl)
    durations = durations.astype(np.float).apply(_divide)

    # Join all df into one
    curr_event_df = curr_event_df.join(onsets)
    curr_event_df = curr_event_df.join(durations)

    # Concat each event_df into one
    new_df = pd.concat([new_df, curr_event_df])

    return new_df


def get_TTL(df, column):
    return float(df.loc[0, column])


def _divide(val):
    return val / 1000


def _subTTL(val, var):
    return (val - var) / 1000


def convert_event_file(
    in_file, in_task, out_file, extract_eprime=False, overwrite=False, verbose=False
):
    """
    Convert txt data from eprime to tsv.

    Parameters
    ----------
    in_file : str
        Task output (.txt) to convert.
    in_task: str
        Config JSON file defining the task you want to convert.
    out_file: str
        output tsv file.

    extract_eprime: bool
        Extract eprime file into raw tsv file. It helps to create a
        config file.
    overwrite: bool
        Force overwriting of the output files.
    verbose: bool
        If set, produces verbose output.

    Returns
    -------
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    if os.path.exists(out_file):
        if not overwrite:
            raise IOError("Output file: {} already exists".format(out_file))

    # Get json file from task name
    task = assert_task_exists(in_task)

    #  Convert eprime txt file to data frame
    df = _text_to_df(in_file)

    if extract_eprime:
        df.to_csv(out_file, index=False, sep="\t")
        return

    # Assert all columns in task exist in df
    assert_task_df(df.columns, task)

    # Final df
    new_df = pd.DataFrame()

    # Get TTL
    TTL = get_TTL(df, task["TTL"])

    for curr_event in task["events"]:
        logging.info("Current event: {}".format(curr_event["name"]))
        new_df = extract_event(df, curr_event, new_df, TTL)

    # Sort new_df by onset
    new_df = new_df.sort_values("onset")
    logging.info(new_df)
    # Extract new_df into tsv file
    new_df.to_csv(out_file, index=False, sep="\t")


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    in_file = args.in_file
    in_task = args.in_task
    out_file = args.out_file
    verbose = args.verbose
    extract_eprime = args.extract_eprime
    overwrite = args.overwrite
    convert_event_file(
        in_file,
        in_task,
        out_file,
        extract_eprime=extract_eprime,
        overwrite=overwrite,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
