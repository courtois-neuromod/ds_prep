import os
import glob
import csv
import retro
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
from retro.scripts.playback_movie import playback_movie


def save_replay_movie(
    bk2_path,
    video_path,
    annotations_path,
    skip_first_step=True,
    inttype=retro.data.Integrations.CUSTOM_ONLY,
):
    """Generate a MP4 movie and an annotation numpy file from a bk2 file.

    Parameters
    ----------
    bk2_path : str
        Path to the bk2 file to replay.
    video_path: str
        Path to the output video file.
    annotations_path: str
        Path to the output annotations file, a numpy file containing the actions and
        info for each frame.
    skip_first_step : bool
        Whether to skip the first step before starting the replay. The intended use of
        retro is to do so (i.e. True) but if the recording was not initiated as
        intended per retro, not skipping (i.e. False) might be required.
        Default is True.
    inttype : retro Integration
        Type of retro integration to use. Default is
        `retro.data.Integrations.CUSTOM_ONLY` for custom integrations, for default
        integrations shipped with retro, use `retro.data.Integrations.STABLE`.
    """
    if video_path[-4:] != ".mp4":
        video_path += ".mp4"
    movie = retro.Movie(bk2_path)
    if skip_first_step:
        movie.step()
    emulator = retro.make(
        movie.get_game(), inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=None
    )
    emulator.initial_state = movie.get_state()
    emulator.reset()
    playback_movie(
        emulator,
        movie,
        video_file=video_path,
        npy_file=annotations_path,
        lossless="mp4",
    )


def process_events_file(event_path, out_dir, int_path):
    """Process an events file, generating the videos and annotations for all the bk2 paths it contains.

    Parameters
    ----------
    event_path : str
        Path of the events file.
    out_dir : str
        Path of the output directory.
    int_path : str
        Path of the game integration directory.
    """
    retro.data.Integrations.add_custom_path(int_path)
    with open(event_path, "r") as f:
        events = csv.DictReader(f, delimiter="\t")
        first_repetition = True
        for row in events:
            if row["stim_file"] not in ("Missing file", "", "n/a"):
                bk2_path = row["stim_file"]
                out_subdir = os.path.join(out_dir, os.path.dirname(bk2_path))
                os.makedirs(out_subdir, exist_ok=True)
                video_path = os.path.join(out_dir, bk2_path[:-3] + "mp4")
                annotations_path = os.path.join(out_dir, bk2_path[:-4])
                if not os.path.exists(video_path):
                    save_replay_movie(
                        os.path.join(args.dataset_path, bk2_path),
                        video_path,
                        annotations_path,
                        skip_first_step=first_repetition,
                    )
            first_repetition = False


def main(args):
    """Scrape the events files of the dataset, and generate video and annotations files
    in the output directory, maintaining the same subdirectories structure as in the
    dataset.
    """
    out_dir = args.dataset_path if args.output_dir is None else args.output_dir
    if args.integration_path is None:
        int_path = os.path.join(args.dataset_path, "stimuli")
    else:
        int_path = args.integration_path
    int_path = os.path.abspath(int_path)
    events_path_template = os.path.join(
        args.dataset_path, "sub-*/ses-*/func/*_events.tsv"
    )
    events_files = glob.glob(events_path_template)
    Parallel(n_jobs=args.n_jobs)(
        delayed(process_events_file)(event_path, out_dir, int_path)
        for event_path in tqdm(events_files, desc="Events files")
    )
    print("======== All done ========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate video replays from bk2 files of a dataset."
    )
    parser.add_argument(
        "-d",
        dest="dataset_path",
        type=str,
        required=True,
        help="Path to the dataset folder.",
    )
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=str,
        default=None,
        help="Path to the output folder, if left to None, <dataset_path> is used, \
            thus the mp4 and npz files will end up next to the bk2 files.",
    )
    parser.add_argument(
        "-i",
        dest="integration_path",
        type=str,
        default=None,
        help="Path to the game integration folder, if left to None, \
            <dataset_path>/stimuli is used.",
    )
    parser.add_argument(
        "-n",
        dest="n_jobs",
        type=int,
        default=-1,
        help="Number of jobs to use for parallelization.",
    )
    args = parser.parse_args()
    main(args)
