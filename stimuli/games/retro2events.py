import os
import retro
import numpy as np
import pandas as pd

SNES_BUTTON_NAMES = [
    "y",
    "a",
    "_",
    "_",
    "up",
    "down",
    "left",
    "right",
    "b",
    "_",
    "_",
    "_",
]


def bk2_to_events(bk2_file, buttons_names=SNES_BUTTON_NAMES, frame_rate=60):
    movie = retro.Movie(bk2_file)

    keys = [[False] * len(SNES_BUTTON_NAMES)]
    while movie.step():
        keys.append([movie.get_key(i, 0) for i in range(len(SNES_BUTTON_NAMES))])
    keys.append([False] * len(SNES_BUTTON_NAMES))
    keys = np.asarray(keys).astype(np.int8)
    key_diff = np.diff(keys, 1, 0)

    events = pd.DataFrame()
    for k, kd in zip(SNES_BUTTON_NAMES, key_diff.T):
        onsets = np.argwhere(kd > 0)[:, 0]
        if len(onsets) == 0:
            continue
        key_events = pd.DataFrame(
            {
                "trial_type": [k] * len(onsets),
                "onset": onsets / frame_rate,
                "duration": (np.argwhere(kd < 0)[:, 0] - onsets) / frame_rate,
            }
        )
        events = events.append(key_events)
    return events.sort_values("onset")


def fill_event_files_from_bk2s(event_file):
    events = pd.read_csv(event_file, delimiter="\t")
    for i, bk2_file in events.iterrows():
        bk2_events = bk2_to_events(os.path.join(bk2_file.stim_file))
        bk2_events.onset += bk2_file.onset
        events = events.append(bk2_events)
    return events.sort_values("onset", ignore_index=True)
