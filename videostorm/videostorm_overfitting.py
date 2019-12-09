"""VideoStorm Overfitting Script."""
import pdb
import argparse
import csv
from video import YoutubeVideo
from videostorm.VideoStorm import VideoStorm

TEMPORAL_SAMPLING_LIST = [20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1]

OFFSET = 0  # The time offset from the start of the video. Unit: seconds


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="VideoStorm with temporal overfitting")
    parser.add_argument("--video", type=str, required=True, help="video name")
    parser.add_argument("--input", type=str, required=True,
                        help="input full model detection file")
    parser.add_argument("--output", type=str, required=True,
                        help="output result file")
    parser.add_argument("--metadata_file", type=str, default=None,
                        # required=True,
                        help="metadata file(json)")
    parser.add_argument("--log", type=str, required=True, help="log file")
    parser.add_argument("--short_video_length", type=int, required=True,
                        help="short video length (seconds)")
    parser.add_argument("--profile_length", type=int, required=True,
                        help="profile length (seconds)")
    parser.add_argument("--frame_rate", type=int, default=None,
                        help="profile length (seconds)")
    args = parser.parse_args()
    return args


def main():
    """VideoStorm."""
    args = parse_args()
    triggered_frames = []
    tstamp = 0
    print("processing", args.video)
    video = YoutubeVideo(args.video, '720p', args.metadata_file, args.input,
                         None, True)
    frame_rate = video.get_frame_rate()
    frame_count = video.get_frame_count()

    system = VideoStorm(TEMPORAL_SAMPLING_LIST, args.log)

    with open(args.output, 'w', 1) as f_out:
        f_out.write("video_name,frame_rate,f1\n")

        # Chop long videos into small chunks
        # Floor division drops the last sequence of frames which is not as
        # long as short_video_length
        profile_frame_cnt = args.profile_length * frame_rate
        chunk_frame_cnt = args.short_video_length * frame_rate
        num_of_chunks = (frame_count-OFFSET*frame_rate)//chunk_frame_cnt

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
            profile_end_frame = profile_start_frame + profile_frame_cnt - 1
            triggered_frames.extend(list(range(profile_start_frame,
                                               profile_end_frame + 1)))

            best_frame_rate = system.profile(clip, video.get_video_detection(),
                                             profile_start_frame,
                                             profile_end_frame,
                                             frame_rate)
            # test on the rest of the short video
            best_sample_rate = frame_rate/best_frame_rate

            # test_start_frame = profile_start_frame
            # test_end_frame = profile_end_frame
            test_start_frame = profile_end_frame + 1
            test_end_frame = end_frame

            f1_score, triggered_frames_tmp = system.evaluate(
                video.get_video_detection(),
                best_sample_rate,
                test_start_frame, test_end_frame)
            triggered_frames.extend(triggered_frames_tmp)

            print(clip, best_frame_rate, f1_score)
            f_out.write(','.join([clip, str(best_frame_rate/frame_rate),
                                  str(f1_score)]) + '\n')

        with open('{}_trace.csv'.format(args.video), 'w', 1) as f_trace:
            writer = csv.writer(f_trace)
            writer.writerow(['frame id', 'timestamp', 'trigger'])
            for i in range(1, frame_count + 1):
                if i in triggered_frames:
                    writer.writerow([i, tstamp, 1])
                else:
                    writer.writerow([i, tstamp, 0])
                tstamp += 1/frame_rate


if __name__ == '__main__':
    main()
