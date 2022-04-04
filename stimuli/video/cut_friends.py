import os, sys, glob
import numpy as np
import pylab as pl

def find_episode_keyframes(movie_file):
    import cv2
    import numpy as np
    credit_start_frames = [np.load(f).astype(np.int32) for f in sorted(glob.glob('credit_start_frame_s[6]_*.npy'))]
    cap = cv2.VideoCapture(movie_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) /2. # deinterlacing
    if len(credit_start_frames) <1:
        cap.release()
        return frame_count, 0
    credit_start_frame_idx = 0
    max_val = 0
    while(cap.isOpened() and credit_start_frame_idx<5400): # should find the frame in the first 2 min
        ret, frame = cap.read()
        for csf_idx, credit_start_frame in enumerate(credit_start_frames) :
            match = (np.square(credit_start_frame-frame)).sum()
            max_val = max(max_val, match)
            if match < np.prod(frame.shape)*64:
                print('found start frame %d at frame %d'%(csf_idx, credit_start_frame_idx))
                break
        else:
            credit_start_frame_idx += 1
            continue
        break
    cap.release()
    if credit_start_frame_idx == 5400:
        print("max: %d/%d)"%(max_val, int(np.prod(frame.shape)*.95)))
        credit_start_frame_idx = -1
    return frame_count, credit_start_frame_idx


def find_episode_keyframes_new(movie_file):
    import cv2
    import numpy as np
    credit_start_frames = [np.load(f).astype(np.uint8) for f in sorted(glob.glob('credit_start_frame_s4*.npy'))]
    cap = cv2.VideoCapture(movie_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))/2. # deinterlacing
    credit_start_frame_idx = -1
    max_val = 0
    while(cap.isOpened() and credit_start_frame_idx<3600): # should find the frame in the first 2 min
        ret, frame = cap.read()
        credit_start_frame_idx += 1
        if frame.sum() < 10:
            continue
        for csf_idx, credit_start_frame in enumerate(credit_start_frames):
            match = cv2.matchTemplate(credit_start_frame, frame, cv2.TM_CCOEFF_NORMED)[0,0]
            max_val = max(max_val, match)
            #print(match)
            if match > .4:
                print('found start frame %d at frame %d: %f'%(csf_idx, credit_start_frame_idx, match))
                break
        else:
            continue

        break
    cap.release()
    if credit_start_frame_idx == 3600:
        print("min: %f"%max_val)
        credit_start_frame_idx = -1
    return frame_count, credit_start_frame_idx

# beware: melt deinterlaces and produces half the input framerate
def cut_friends_episode(movie_file, episode_length,
                        n_segments, segment_name, output_file,
                        credit_start, credit_duration=1344,
                        overlap = 6*30,
                        fade_in=2*30, fade_out=2*30, black_screen_end=6*30):
    f = open(output_file, 'w')
    start = 0
    seg_length = (episode_length-credit_duration)/n_segments
    stop = seg_length + credit_duration

    print(stop-start-credit_duration+overlap+black_screen_end)

    base_cmd = """
singularity run \
    -B $PWD:/input \
    -B $PWD:/output \
    /data/neuromod/containers/melt.simg \
    -silent """


    movie_file = os.path.join('/input', movie_file)

    # the first segment, we remove the opening credit/song... and replace it with a black screen of the duration equal to overlap between runs
    # this way it is easy to make all segments of the same length

    output_seg_fname = os.path.join('/output',segment_name % 'a')
    seg_idx = 1
    stop = 0
    if credit_start > 0:
        f.write(base_cmd +
 """colour:black out=%d \
    %s in=%d out=%d -mix %d -mixer luma \
    colour:black out=%d -mix %d -mixer luma \
    %s in=%d out=%d -mix %d -mixer luma \
    colour:black out=%d -mix %d -mixer luma \
    -attach-track ladspa.1403 0=-25 1=0.25 2=0.4 3=0.6 \
    -attach-track ladspa.1913 0=17 1=-3 2=0.5 \
    -attach-track volume:-70db end=0db in=0 out=%d \
    -attach-track volume:0db end=-70db in=%d out=%d \
    -consumer avformat:%s f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=1500k
    """%(fade_in-1, # duration of fade_in from black
         movie_file, start, credit_start, fade_in, # segment start, credit start
         fade_out+overlap+fade_in, fade_out-1, # fade_out and black screen
         movie_file, credit_start+credit_duration, stop, fade_in, # credit stop, segment stop
         fade_out+black_screen_end, fade_out-1, # fade_out and black screen
         fade_in, stop-start-fade_out,stop-start, # sound fade-in and fade-out
         output_seg_fname))
        seg_idx=2

    for seg_idx in range(seg_idx, n_segments+1):
        seg_letter = 'abcdefghijkl'[seg_idx-1]
        start = max(0, stop - overlap)
        stop += seg_length
        print(stop-start+black_screen_end)
        output_seg_fname = os.path.join('/output',segment_name % seg_letter)
        f.write(base_cmd +
    """colour:black out=%d \
    %s in=%d out=%d -mix %d -mixer luma \
    colour:black out=%d -mix %d -mixer luma \
    -attach-track ladspa.1403 0=-25 1=0.25 2=0.4 3=0.6 \
    -attach-track ladspa.1913 0=17 1=-3 2=0.5 \
    -attach-track volume:-70db end=0db in=0 out=%d \
    -attach-track volume:0db end=-70db in=%d out=%d \
    -consumer avformat:%s f=matroska acodec=libmp3lame ab=256k vcodec=libx264 b=1500k
        """%(fade_in-1, # duration of fade_in from black
             movie_file, start, stop, fade_in, # segment start, credit start
             (fade_out+black_screen_end), fade_out-1, # fade_out and black screen
             fade_in, stop-start-fade_out, stop-start, # sound fade-in and fade-out
             output_seg_fname))

    f.close()



def cut_all_files(path):

    for input_file in sorted(glob.glob(os.path.join(path,'friends_s??e??.mkv'))+glob.glob(os.path.join(path,'friends_s??e??_e??.mkv'))):
        basename = os.path.basename(input_file)
        print('processing %s'%basename[:-4])
        episode_length, credit_start_frame_idx = find_episode_keyframes(input_file)
        print(episode_length, credit_start_frame_idx)
        if credit_start_frame_idx == -1:
            print('WARNING: the opening credit frame was not found, we will assume credit start from the beginning')
            credit_start_frame_idx = 0
        # cut in segments of 10+min
        number_of_segments = int(np.floor(episode_length/30/600))
        print(f"cutting {number_of_segments} segments")
        cut_friends_episode(
            input_file,
            episode_length,
            number_of_segments,
            os.path.join(path, basename[:-4] + '%s.mkv'),
            os.path.join(path, 'cut_%s.sh'%basename[:-4]),
            credit_start_frame_idx)

if __name__== "__main__":
    cut_all_files(sys.argv[1])
