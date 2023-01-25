import os, sys
import numpy as np
import pandas as pd
import retro
import tqdm

stop_tags = ["VideoGame", "stopped at"]
rep_tag = "level step: 0"
TTL_tag = "fMRI TTL 0"
record_tag = "VideoGame: recording movie"
abort_tags = ["<class 'src.tasks.videogame.VideoGameMultiLevel'>", "abort"]
restart_tags= ["<class 'src.tasks.videogame.VideoGameMultiLevel'>", "restart"]
complete_tags = [" <class 'src.tasks.videogame.VideoGameMultiLevel'>", "complete"]

retro.data.Integrations.add_custom_path(
        os.path.join(os.getcwd(), "stimuli")
)


def load_log_file(fname):
    return np.loadtxt(
        fname,
        delimiter="\t",
        dtype=dict(
            names=("time", "event_type", "event"), formats=(np.float, "U4", "U255")
        ),
        converters={0: float, 1: lambda x: x.strip(), 2: lambda x: x.strip()},
    )

def split_rep_logs(log):
    rep_logs = []
    rep_log = None
    TTL = None
    for e in log:
        if not rep_log is None:
            rep_log.append(e)
        if record_tag in e[2]:
            if rep_log is not None:
                rep_logs.append((record,rep_log))
            rep_log = []
            record = "/".join(e[2].split(" ")[-1].split("/")[-4:])
            print(f"\n\n\n{record}")
        elif all(stt in e[2] for stt in stop_tags):
            stop = e
        elif all(abt in e[2] for abt in abort_tags) or all(abt in e[2] for abt in restart_tags):
            TTL = None
            rep_log = None
    return rep_logs

def logs2event_files(in_files, out_file_tpl):
    repetition = []
    run = 0
    TTL = None
    last_rep_last_run = 'aaa'
    for in_file in sorted(in_files):
        rep_log = None
        log = load_log_file(in_file)
        for e in log:
            if not rep_log is None:
                rep_log.append(e)
            if record_tag in e[2]:
                record = "/".join(e[2].split(" ")[-1].split("/")[-4:])
#                print(f"\n{record}")
            elif e[2] == TTL_tag:
                if not TTL:
                    run += 1  # only increment if the previous scan was not aborted
                    TTL = e[0]
                rep_log = []
                print('##########################################')
                repetition.append([])
            elif e[2] == rep_tag:
                rep = e
            elif all(stt in e[2] for stt in stop_tags):
                stop = e
                print(stop)

                if TTL:
                    record = record.replace('sourcedata','sourcedata/behavior').replace('ses-0','ses-shinobi_0')
                    bk2 = retro.Movie(record)
                    bk2_dur  = 0
                    while bk2.step():
                        bk2_dur += 1

                    duration_log = stop[0] - rep[0]
                    if np.abs(duration_log-bk2_dur/fps) > .05 or record == last_rep_last_run:
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
                        duration_steps = int(duration_log*fps)
                        keypresses, rewards = extract_keypress_rewards(rep_log)
                        print(keypresses, rewards)
                        kp_1hot = keypresses_1hot(keypresses, duration_steps)
                        bk2 = retro.Movie(record)
                        print(find_keyreleases(bk2, kp_1hot, rewards, duration_steps))
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
                rep_log = None
#                print(repetition[-1])
            elif all(cpt in e[2] for cpt in complete_tags):
                TTL = None
                last_rep_last_run = repetition[-1][-1]
            elif all(abt in e[2] for abt in abort_tags) or all(abt in e[2] for abt in restart_tags):
                if TTL and len(repetition): # only if TTL was received
                    del repetition[-1]
                if not all(abt in e[2] for abt in restart_tags):
                    TTL = None
                rep_log = None
        TTL = None  # reset the TTL
    run = 0
    print(repetition)
    for reps in repetition:
        if len(reps) == 0:  # skip empty tasks or without scanning
            print("empty task")
            continue
        run += 1
        out_file = out_file_tpl % run
        df = pd.DataFrame(
            reps, columns=["trial_type", "onset", "duration", "duration_bk2", "level", "stim_file"]
        )
        df.to_csv(out_file, sep="\t", index=False)

fps=60.15

def extract_keypress_rewards(rep_log):
    keypresses = {}
    rewards = {}
    refs = {}
    for e in rep_log:
        if 'level step:' in e[2]:
            ref_time = e[0]
            step = int(e[2].split(' ')[2])
            refs[ref_time] = step
    #ref_keys = np.asarray(list(refs.keys()))
    #for e in rep_log:
            closest_ref = ref_time
        if 'Keypress' in e[2]:
            key = e[2].split(' ')[1]
            if key in 'rludyab':

                keypress_step = refs[closest_ref] + int(np.ceil((e[0]-closest_ref)*fps+1/fps))
                if keypress_step in keypresses:
                    keypresses[keypress_step].append(key)
                else:
                    keypresses[keypress_step] = [key]
        elif 'Reward' in e[2]:
            reward = float(e[2].split(' ')[1])
            reward_step = refs[closest_ref] + int(np.round((e[0]-closest_ref)*fps)) + 1
            if reward_step in rewards:
                print(f"error reward double assigned {reward_step} {reward}")
                if reward_step - 1 in rewards:
                    print(f"error cannot retro assign {reward_step-1} {reward}")
                else:
                    rewards[reward_step-1] = rewards[reward_step]
            rewards[reward_step] = reward
    return keypresses, rewards



KEY_SET = ["y", "a", "_", "_", "u", "d", "l", "r", "b", "_", "_", "_"]


def keypresses_1hot(keypresses, duration):
    hot1 = np.zeros((duration, len(KEY_SET)), dtype=np.bool)
    for idx, ks in keypresses.items():
        for k in ks:
            hot1[idx, KEY_SET.index(k)] = True
    return hot1

MAX_PRESS_LENGTH = 1200

def _rec_find_keyreleases(
        env, initial_step, key_state, keypresses,
        rewards, key_releases, duration,
        depth=1, total_rewards=0, render=False):
    key_state = np.logical_or(key_state, keypresses[initial_step])
    keys = [KEY_SET[i] for i,p in enumerate(keypresses[initial_step]) if p]
    long_keys = [k for k in keys if k in "lrdby"] # keys with press duration effect:

    step_reward = total_rewards
    _obs, _rew, done, _info = env.step(key_state)
    if render and initial_step%render == 0:
        env.render()
    if _rew:
        print(f"reward at {initial_step: }: {step_reward}")
    if initial_step in rewards:
        if rewards[initial_step] != step_reward or _rew==0:
            return -1, step_reward # kill that branch
        else:
            print(f"___ match reward at {initial_step}= {rewards[initial_step]}")

    if not len(long_keys):
        # release key as keypress duration has no impact
        key_state[keypresses[initial_step]] = False

        step = initial_step+1
        # loop when no long keypress to limit recursion level
        while not any(keypresses[step]):
            _obs, _rew, done, _info = env.step(key_state)
            if render and step%render == 0:
                env.render()
            step_reward += _rew
            if _rew:
                print(f"reward at {step: }: {step_reward}")
            if step in rewards:
                if rewards[step] != step_reward or _rew==0:
                    return 0, step_reward # kill that branch, but allow backtrack
                else:
                    print(f"### match reward at {step}= {rewards[step]}")
            step += 1
        return _rec_find_keyreleases(
            env, step, key_state,
            keypresses, rewards, key_releases,
            duration, depth+1, step_reward, render=render
        )
    else:
        # find next occurence of press (which means it has been released before that)
        max_press_length = np.where(keypresses[initial_step+1:, keypresses[initial_step]])[0]
        max_press_length = max_press_length[0]-2 if len(max_press_length) else min(MAX_PRESS_LENGTH, duration-initial_step-1)

        backtrack_state = env.em.get_state()
        backtrack_key_state = np.copy(key_state)

        for key_len in tqdm.tqdm(
            range(1, max_press_length),
            desc=f"depth: {depth}: keys: {keys}, step: {initial_step}, reward: {step_reward}",
            leave=depth < 5):
            step = initial_step + key_len

            # do key release on next step
            key_state[keypresses[initial_step]] = False

            ret, step_reward =_rec_find_keyreleases(
                env, step, key_state,
                keypresses, rewards, key_releases,
                duration, depth+1, step_reward, render=render
            )
            if ret < 0:
                return 0, step_reward # kill branch, allow backtrack one level down
            elif ret < 1:
                # backtrack one step
                env.initial_state = backtrack_state
                env.reset()
                # and move one step with key still pressed
                key_state = np.copy(backtrack_key_state)
                _obs, _rew, done, _info = env.step(key_state)
                step_reward += _rew
                if step in rewards:
                    if rewards[step] != step_reward or _rew==0:
                        return -1, step_reward # kill that branch
                    else:
                        print(f"$$$ match reward at {step}= {rewards[step]}")
                backtrack_state = env.em.get_state()
            else:
                return ret, step_reward
    return False, step_reward


def find_keyreleases(movie, keypresses, rewards, duration, render=False):

    #reward_steps = np.asarray(list(rewards.keys()))
    key_releases = {}
    total_rewards = 0
    keypresses[0]

    try:
        env = retro.make(
            game=movie.get_game(),
            state=None,
            players=movie.players,
            scenario='scenario',
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )
        #movie.step()

        env.initial_state = movie.get_state()
        env.reset()

        ret, ret_rewards = _rec_find_keyreleases(
            env, 0, keypresses[0], keypresses, rewards, key_releases, duration, 0, total_rewards, render=render,
        )
        if ret:
            print(f" solution found")
            key_releases[step] = ret
            total_rewards = ret_rewards
        else:
            raise RuntimeError(f"no solution found")

        print(key_releases)
        return key_releases
    finally:
        env.render(close=True)
        env.close()
