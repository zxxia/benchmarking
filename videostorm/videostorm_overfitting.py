""" VideoStorm Overfitting Script """
import argparse
from utils.model_utils import load_full_model_detection, \
    filter_video_detections
from utils.utils import load_metadata
from videostorm.profiler import profile, profile_eval
from constants import CAMERA_TYPES
TEMPORAL_SAMPLING_LIST = [20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1]

OFFSET = 0  # The time offset from the start of the video. Unit: seconds


def parse_args():
    """ parse arguments """
    parser = argparse.ArgumentParser(
        description="VideoStorm with temporal overfitting")
    parser.add_argument("--video", type=str, required=True, help="video name")
    parser.add_argument("--input", type=str, required=True,
                        help="input full model detection file")
    parser.add_argument("--output", type=str, required=True,
                        help="output result file")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="metadata file(json)")
    parser.add_argument("--log", type=str, required=True, help="log file")
    parser.add_argument("--short_video_length", type=int, required=True,
                        help="short video length (seconds)")
    parser.add_argument("--profile_length", type=int, required=True,
                        help="profile length (seconds)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    f_log = open(args.log, 'w', 1)
    f_log.write("video_name,frame_rate,f1\n")
    with open(args.output, 'w', 1) as f_out:
        f_out.write("video_name,frame_rate,f1\n")
        print("processing", args.video)
        metadata = load_metadata(args.metadata_file)
        frame_rate = metadata['frame rate']

        gtruth, num_of_frames = load_full_model_detection(args.input)

        gtruth = filter_video_detections(gtruth, target_types={3, 8})
        # Filter ground truth if it is static camera
        if args.video in CAMERA_TYPES['static']:
            gtruth = filter_video_detections(gtruth, width_range=(0, 1280/2),
                                             height_range=(0, 720/2))
        gtruth = filter_video_detections(gtruth, height_range=(720//20, 720))

        # only for road_trip to remove the bboxes on the operation deck
        if args.video == 'road_trip':
            for frame_idx in gtruth:
                tmp_boxes = []
                for box in gtruth[frame_idx]:
                    xmin, ymin, xmax, ymax = box[:4]
                    if ymin >= 500 and ymax >= 500:
                        continue
                    if (xmax - xmin) >= 2/3 * 1280:
                        continue
                    tmp_boxes.append(box)
                gtruth[frame_idx] = tmp_boxes

        # Chop long videos into small chunks
        # Floor division drops the last sequence of frames which is not as
        # long as short_video_length
        profile_frame_cnt = args.profile_length * frame_rate
        chunk_frame_cnt = args.short_video_length * frame_rate
        num_of_chunks = (num_of_frames-OFFSET*frame_rate)//chunk_frame_cnt

        for i in range(num_of_chunks):
            clip = args.video + '_' + str(i)
            # the 1st frame in the chunk
            start_frame = i * chunk_frame_cnt + 1 + OFFSET * frame_rate
            # the last frame in the chunk
            end_frame = (i + 1) * chunk_frame_cnt + OFFSET * frame_rate
            print('short video start={}, end={}'.format(start_frame,
                                                        end_frame))
            assert args.short_video_length >= args.profile_length

            profile_start_frame = start_frame
            profile_end_frame = profile_start_frame+profile_frame_cnt-1

            best_frame_rate = profile(clip, gtruth, gtruth,
                                      profile_start_frame, profile_end_frame,
                                      frame_rate, TEMPORAL_SAMPLING_LIST,
                                      f_log)
            # test on the rest of the short video
            best_sample_rate = frame_rate/best_frame_rate

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
