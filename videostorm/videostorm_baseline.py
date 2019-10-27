""" VideoStorm baseline Script """
import argparse
import numpy as np
from utils.model_utils import load_full_model_detection, \
    filter_video_detections
from utils.utils import load_metadata
from videostorm.profiler import profile, profile_eval
from constants import CAMERA_TYPES
PATH = '/data/zxxia/videos/'
CSV_PATH = '/data/zxxia/benchmarking/results/videos/'
TEMPORAL_SAMPLING_LIST = [20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1]
TARGET_F1 = 0.9
OFFSET = 0  # The time offset from the start of the video. Unit: seconds


def main():
    parser = argparse.ArgumentParser(description="VideoStorm baseline")
    parser.add_argument("--video", type=str, required=True, help="video name")
    parser.add_argument("--input", type=str, required=True,
                        help="input full model detection file")
    parser.add_argument("--output", type=str, required=True,
                        help="output result file")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="metadata file(json)")
    parser.add_argument("--log", type=str, required=True, help="log file")
    parser.add_argument("--sample_rate", type=int, required=True,
                        help="sample two frames every n frame")
    parser.add_argument("--short_video_length", type=int, required=True,
                        help="short video length (seconds)")
    parser.add_argument("--profile_length", type=int, required=True,
                        help="profile length (seconds)")
    args = parser.parse_args()
    dataset = args.video
    output_file = args.output
    input_file = args.input
    log_file = args.log
    sample_rate = args.sample_rate
    short_video_length = args.short_video_length
    profile_length = args.profile_length

    f_log = open(log_file, 'w', 1)
    f_log.write("video_name,frame_rate,f1\n")
    with open(output_file, 'w', 1) as f_out:
        f_out.write("video_name,frame_rate,f1\n")
        print("processing", dataset)
        metadata = load_metadata(args.metadata_file)
        frame_rate = metadata['frame rate']

        gtruth, nb_frames = load_full_model_detection(input_file)

        gtruth = filter_video_detections(gtruth, target_types={3, 8})
        # Filter ground truth if it is static camera
        if dataset in CAMERA_TYPES['static']:
            gtruth = filter_video_detections(gtruth, width_range=(0, 1280/2),
                                             height_range=(0, 720/2))
        gtruth = filter_video_detections(gtruth, height_range=(720//20, 720))
        # only for road_trip
        for frame_idx in gtruth:
            tmp_boxes = []
            for box in gtruth[frame_idx]:
                xmin, ymin, xmax, ymax = box[:4]
                if ymin >= 500 and ymax >= 500 and (xmax - xmin) >= 1280/2:
                    continue
                tmp_boxes.append(box)
            gtruth[frame_idx] = tmp_boxes

        # Chop long videos into small chunks
        # Floor division drops the last sequence of frames which is not as
        # long as short_video_length
        profile_frame_cnt = profile_length * frame_rate
        chunk_frame_cnt = short_video_length * frame_rate
        num_of_chunks = (nb_frames-OFFSET*frame_rate)//chunk_frame_cnt

        nb_frames_sampled = nb_frames // sample_rate * 2
        nb_svid_sampled = nb_frames_sampled // chunk_frame_cnt
        print(nb_frames, sample_rate, nb_frames_sampled,
              nb_svid_sampled, chunk_frame_cnt)
        profile_start_frames = np.random.randint(1,
                                                 nb_frames-profile_frame_cnt,
                                                 size=nb_svid_sampled)
        profile_end_frames = profile_start_frames + profile_frame_cnt
        print(profile_start_frames)
        print(profile_end_frames)

        assert profile_end_frames.all() <= nb_frames
        best_sample_rates = list()
        for profile_start_frame, profile_end_frame in zip(profile_start_frames,
                                                          profile_end_frames):
            clip = dataset+'_'+str(profile_start_frame//profile_frame_cnt)
            print('profile {}-{}'
                  .format(profile_start_frame, profile_end_frame))
            best_frame_rate = profile(clip, gtruth, gtruth,
                                      profile_start_frame,
                                      profile_end_frame, frame_rate,
                                      TEMPORAL_SAMPLING_LIST, f_log)
            # test on the rest of the short video
            best_sample_rates.append(frame_rate / best_frame_rate)
        print(best_sample_rates)
        for i in range(num_of_chunks):
            clip = dataset + '_' + str(i)
            # the 1st frame in the chunk
            start_frame = i * chunk_frame_cnt + 1 + OFFSET * frame_rate
            # the last frame in the chunk
            end_frame = (i + 1) * chunk_frame_cnt + OFFSET * frame_rate
            print('short video start={}, end={}'.format(start_frame,
                                                        end_frame))
            # profile the first PROFILE_LENGTH seconds of the chunk
            assert short_video_length >= profile_length

            profile_start_frame = start_frame
            profile_end_frame = profile_start_frame+profile_frame_cnt-1

            # test on the rest of the short video
            best_sample_rate_idx = i//(num_of_chunks//len(best_sample_rates))
            if best_sample_rate_idx >= len(best_sample_rates):
                best_sample_rate_idx = len(best_sample_rates) - 1
            print(len(best_sample_rates), best_sample_rate_idx)
            best_sample_rate = best_sample_rates[best_sample_rate_idx]

            test_start_frame = profile_start_frame
            test_end_frame = profile_end_frame

            f1_score = profile_eval(gtruth, gtruth, best_sample_rate,
                                    test_start_frame, test_end_frame)

            print(clip, best_frame_rate, f1_score)
            f_out.write(','.join([clip, str(best_frame_rate/frame_rate),
                                  str(f1_score)])+'\n')
    f_log.close()


if __name__ == '__main__':
    main()
