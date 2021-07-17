import os, sys
import numpy as np
import pandas as pd
import retro

stop_tags = ["VideoGame", "stopped at"]
rep_tag = "level step: 0"
TTL_tag = "fMRI TTL 0"
record_tag = "VideoGame: recording movie"
abort_tags = ["<class 'src.tasks.videogame.VideoGameMultiLevel'>", "abort"]
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
        elif all(abt in e[2] for abt in abort_tags):
            TTL = None
            rep_log = None
    return rep_logs

def logs2event_files(in_files, out_file_tpl):
    repetition = []
    run = 0
    TTL = None
    last_rep_last_run = 'aaa'
    for in_file in in_files:
        rep_log = None
        log = load_log_file(in_file)
        for e in log:
            if not rep_log is None:
                rep_log.append(e)
            if record_tag in e[2]:
                record = "/".join(e[2].split(" ")[-1].split("/")[-4:])
                print(f"\n\n\n{record}")
            elif e[2] == TTL_tag:
                if not TTL:
                    run += 1  # only increment if the previous scan was not aborted
                    TTL = e[0]
                rep_log = []
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
                        duration_steps = int(duration_log*60.02)
                        keypresses, rewards = extract_keypress_rewards(rep_log)
                        #print(keypresses, rewards)
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
            elif all(cpt in e[2] for cpt in complete_tags):
                TTL = None
                last_rep_last_run = repetition[-1][-1]
            elif all(abt in e[2] for abt in abort_tags):
                TTL = None
                if len(repetition): # only if TTL was received
                    del repetition[-1]
                rep_log = None
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
    refs = {}
    for e in rep_log:
        if 'level step:' in e[2]:
            refs[e[0]] = int(e[2].split(' ')[2])
    ref_keys = np.asarray(list(refs.keys()))
    for e in rep_log:
        closest_ref = ref_keys[np.argmin(np.abs(ref_keys-e[0]))]
        if 'Keypress' in e[2]:
            key = e[2].split(' ')[1]
            if key in 'rludyab':
                
                keypress_step = refs[closest_ref] + int(np.ceil((e[0]-closest_ref)*60))
                if keypress_step in keypresses:
                    keypresses[keypress_step].append(key)
                else:
                    keypresses[keypress_step] = [key]
        elif 'Reward' in e[2]:
            reward = float(e[2].split(' ')[1])
            reward_step = refs[closest_ref] + int(np.floor((e[0]-closest_ref)*60))
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

def _rec_find_keyreleases(env, initial_step, key_state, keypresses, rewards, key_releases, duration, depth=1, total_rewards=0):
    key_state = np.logical_or(key_state, keypresses[initial_step])
    keys = [KEY_SET[i] for i,p in enumerate(keypresses[initial_step]) if p]
    long_keys = [k for k in keys if k in "lrdby"] # key with press duration effect:

    _obs, _rew, done, _info = env.step(key_state)
    if initial_step%60 == 0:
        env.render()
    if _rew:
        total_rewards += _rew
    if initial_step in rewards:
        #print(f"\n### evaluate reward {_rew} {rewards[initial_step]}\n")
        if rewards[initial_step] != total_rewards or _rew==0:
            sys.stdout.write("#"*depth + f" reward mismatch: step :{initial_step}, depth: {depth}, {total_rewards} != {rewards[initial_step]} \n")
            sys.stdout.flush()
            return False # kill that branch
        else:
            print("\n" + "#"*depth +" reward match: step :{initial_step}, depth: {depth}\n")

    
    if not len(long_keys):
        # reset key as length has no impact
        key_state[keypresses[initial_step]] = False
    else:
        # find next occurence of press (which means it has been released before that)
        max_press_length = np.where(keypresses[initial_step+1:, keypresses[initial_step]])[0]
        max_press_length = max_press_length[0]-1 if len(max_press_length) else min(MAX_PRESS_LENGTH, duration-initial_step-1)

        backtrace_state = env.em.get_state()
        backtrace_key_state = key_state

        sys.stdout.write("\n")

        for key_len in range(1, max_press_length):
            sys.stdout.write("#"*depth + f" keys: {keys}, step: {initial_step}, depth: {depth}, key_len: {key_len}/{max_press_length}    \r")
            sys.stdout.flush()
            if initial_step+key_len >= duration:
                return True # reached the end of the run

            # do key release on next step
            key_state = np.logical_xor(
                key_state,
                keypresses[initial_step]) # key release # todo:fix

            for step_eval in range(initial_step+key_len, duration):
                ret =_rec_find_keyreleases(
                    env, step_eval, key_state, keypresses, rewards, key_releases, duration, depth+1, total_rewards,
                )
                if ret is False:
                    break
            if ret is False:
                # backtrack one step
                env.initial_state = backtrace_state
                env.reset()
                # and move one step with key still pressed
                key_state = np.logical_and(backtrace_key_state, keypresses[initial_step+key_len])
                _obs, _rew, done, _info = env.step(key_state)
                #env.render()
                backtrace_state = env.em.get_state()

                # if not releasing does not match rewards, abandon the whole search of that branch
                step_backtrack = initial_step+key_len
                if step_backtrack in rewards:
                    if rewards[step_backtrack] != total_rewards+_rew or _rew==0:
                        print("#"*depth + f" abort: step :{step_backtrack}, depth: {depth}, key_len: {key_len}/{max_press_length}     ")
                        return False # kill that branch
            else:
                return ret
        print(f"\n")


def find_keyreleases(movie, keypresses, rewards, duration):

    #reward_steps = np.asarray(list(rewards.keys()))
    key_releases = {}
    
    key_state = keypresses[0]

    try:
        env = retro.make(
            game=movie.get_game(),
            state=None,
            players=movie.players,
            inttype=retro.data.Integrations.CUSTOM_ONLY,
        )
        movie.step()
        
        env.initial_state = movie.get_state()
        env.reset()

        step = 1
        while step < duration:
            if any(keypresses[step]):
                ret = _rec_find_keyreleases(
                    env, step, key_state, keypresses, rewards, key_releases, duration
                )
                if ret:
                    print(f" solution found for step {step}")
                else:
                    raise RuntimeError(f"no solution found for step {step}")
                step += 1
            else:
                _obs, _rew, done, _info = env.step(key_state)
                step += 1

        print(key_releases)
        return key_releases
    finally:
        env.render(close=True)
        env.close()
