import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

tr = 1.49
delay = 3.0


def align_transcript(tr: float, json_path: str, tsv_dir: str) -> None:
    """Converts word timestamps into TR aligned to .tsv file.

    Args:
        tr: repetition time of fMRI scan.
        json_path: Path to timestamp json files.
        tsv_dir: Path to the output TR alignment files
    """
    tsv_dict: dict[str, list[Any]] = {
        "words_per_tr": [],
        "onsets_per_tr": [],
        "durations_per_tr": [],
    }

    with open(json_path) as json_file:
        j_file = json.load(json_file)
        segment_path = Path(json_path)
        segment_name = segment_path.stem

        #  list of word, onset, offset, and confidence
        words = j_file["results"]["channels"][0]["alternatives"][0]["words"]

        # lists the boundaries of tr windows that lasts until,
        # the last stimuli offset + delay
        tr_boundaries = list(np.arange(0, words[-1]["offset"] + delay, tr))

        # loops over the tr windows and find the words fall in
        index = 0
        for tr_onset in tr_boundaries:
            tr_words = []
            tr_onsets = []
            tr_durations = []

            while (
                index < len(words) and words[index]["offset"] < tr_onset + tr
            ):
                tr_words.append(words[index]["word"])
                tr_onsets.append(words[index]["onset"])
                tr_durations.append(
                    words[index]["offset"] - words[index]["onset"],
                )
                index += 1

            tsv_dict["words_per_tr"].append(tr_words)
            tsv_dict["onsets_per_tr"].append(tr_onsets)
            tsv_dict["durations_per_tr"].append(tr_durations)

        df_results_rec = pd.DataFrame.from_dict(tsv_dict)

        df_results_rec.insert(loc=0, column="segment", value=segment_name)

        # for AssemblyAI speech-to-text output
        tsv_file = Path(tsv_dir) / f"{segment_name}.tsv"
        df_results_rec.to_csv(tsv_file, sep="\t", index=False)


def parse_arguments() -> Namespace:
    """Function to parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Align words with the fmri repetition time.",
    )
    parser.add_argument("--data_dir", help="where the time_stamps files are")
    parser.add_argument("--stimuli_name", help="name of the stimuli folder")
    return parser.parse_args()


def main() -> None:
    """."""
    args = parse_arguments()
    json_dir = Path(args.data_dir) / "word_timestamps" / args.stimuli_name
    json_files = sorted(json_dir.glob("*.json"))

    tsv_dir = Path(args.data_dir) / "tr_alignment" / args.stimuli_name
    tsv_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over the json files for TR alignment
    for json_path in json_files:
        align_transcript(tr, json_path, tsv_dir)


if __name__ == "__main__":
    main()
