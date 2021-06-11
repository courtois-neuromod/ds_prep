import numpy as np
import pandas as pd
import retro

stop_tags = ["VideoGame", "stopped at"]
rep_tag = "level step: 0"
TTL_tag = "fMRI TTL 0"
record_tag = "VideoGame: recording movie"
abort_tags = ["<class 'src.tasks.videogame.VideoGameMultiLevel'>", "abort"]
complete_tags = [" <class 'src.tasks.videogame.VideoGameMultiLevel'>", "complete"]


def logs2event_files(in_files, out_file_tpl):
    repetition = []
    run = 0
    TTL = None
    last_rep_last_run = 'aaa'
    rep_log = None
    for in_file in in_files:
        log = np.loadtxt(
            in_file,
            delimiter="\t",
            dtype=dict(
                names=("time", "event_type", "event"), formats=(np.float, "U4", "U255")
            ),
            converters={0: float, 1: lambda x: x.strip(), 2: lambda x: x.strip()},
        )
        for e in log:
            if not rep_log is None:
                rep_log.append(e)

            if record_tag in e[2]:
                record = "/".join(e[2].split(" ")[-1].split("/")[-4:])
                rep_log = []
            elif e[2] == TTL_tag:
                if not TTL:
                    run += 1  # only increment if the previous scan was not aborted
                TTL = e[0]
                repetition.append([])
            elif e[2] == rep_tag:
                rep = e
            elif all(stt in e[2] for stt in stop_tags):
                stop = e
                if TTL:
                    bk2 = retro.Movie(record)
                    bk2_dur  = 0
                    while bk2.step():
                        bk2_dur += 1

                    duration_log = stop[0] - rep[0]
                    if np.abs(duration_log-bk2_dur/60.) > 1 or record == last_rep_last_run:
                        print(f"error : run-{len(repetition)} {rep[0]} {record} {duration_log} - {bk2_dur/60.}={duration_log-(bk2_dur/60.)}")
                        repetition[-1].append(
                            (
                                "gym-retro_game",
                                rep[0] - TTL,
                                duration_log,
                                bk2_dur/60.,
                                record.split('Level')[-1][:3],
                                None,
                            )
                        )
                        keypresses, rewards = extract_keypress_rewards(rep_log)
                        print(keypresses, rewards)
                    else:
                        repetition[-1].append(
                            (
                                "gym-retro_game",
                                rep[0] - TTL,
                                duration_log,
                                bk2_dur/60.,
                                record.split('Level')[-1][:3],
                                record,
                            )
                        )
            elif all(cpt in e[2] for cpt in complete_tags):
                TTL = None
                last_rep_last_run = repetition[-1][-1]
            elif all(abt in e[2] for abt in abort_tags):
                TTL = None
                if len(repetition): # only if TTL was received
                    del repetition[-1]
        TTL = None  # reset the TTL
    run = 0
    for reps in repetition:
        if len(reps) == 0:  # skip empty tasks or without scanning
            continue
        run += 1
        out_file = out_file_tpl % run
        df = pd.DataFrame(
            reps, columns=["trial_type", "onset", "duration", "duration_bk2", "level", "stim_file"]
        )
        df.to_csv(out_file, sep="\t", index=False)



def extract_keypress_rewards(rep_log):
    keypresses = {}
    rewards = {}
    for e in rep_log:
        if 'level step:' in e[2]:
            ref = (e[0], int(e[2].split(' ')[2]))
        elif 'Keypress' in e[2]:
            key = e[2].split(' ')[1]
            if key in 'rludyab':
                keypress_step = ref[1] + int(np.ceil((e[0]-ref[0])/60.))
                keypresses[keypress_step] = key
        elif 'Reward' in e[2]:
            reward = float(e[2].split(' ')[1])
            reward_step = ref[1] + int(np.ceil((e[0]-ref[0])/60.))
            rewards[reward_step] = reward
    return keypresses, rewards
