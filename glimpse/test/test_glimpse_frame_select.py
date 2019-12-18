import argparse
import csv
import numpy as np
from glimpse.Glimpse import Glimpse
from video import YoutubeVideo

PARA1_LIST_DICT = {
    'crossroad': np.arange(40, 52, 2),
    'crossroad2': np.arange(30, 42, 2),
    'crossroad3': np.arange(80, 100, 3),
    'crossroad4': np.arange(40, 82, 2),
    'drift': np.arange(290, 400, 10),
    'driving1': np.arange(15, 25, 2),
    'driving2': np.arange(10, 30, 2),
    'driving_downtown': np.arange(4, 20, 2),
    'highway': np.arange(35, 40, 1),
    'highway_normal_traffic': np.arange(34, 40, 2),
    'jp': np.arange(30, 40, 2),
    'jp_hw': np.arange(30, 40, 2),
    'lane_split': np.arange(6, 14, 2),
    'motorway': np.arange(2, 6, 0.5),
    'nyc': np.arange(2, 22, 2),
    'park': np.arange(2, 20, 2),
    'russia': np.arange(100, 400, 20),
    'russia1': np.arange(100, 400, 20),
    'traffic': np.arange(6, 15, 1),
    'tw': np.arange(25, 80, 5),
    'tw1': np.arange(25, 80, 5),
    # 'tw_road': np.arange(15, 45, 5),
    'tw_under_bridge': np.arange(350, 450, 10),
}
PARA2_LIST = [50, 30, 20, 10]


def parse_args():
    """Parse the input arguments required by the code."""
    parser = argparse.ArgumentParser(
        description="Glimpse with perfect tracking")
    # parser.add_argument("--path", type=str, required=True,
    #                     help="path contains all datasets")
    parser.add_argument("--video", type=str, required=True,
                        help="video name")
    parser.add_argument("--metadata", type=str,  # required=True,
                        help="metadata file in Json")
    parser.add_argument("--output", type=str, required=True,
                        help="output result file")
    parser.add_argument("--log", type=str,
                        help="profiling log file")
    parser.add_argument("--short_video_length", type=int, required=True,
                        help="short video length in seconds")
    parser.add_argument("--profile_length", type=int, required=True,
                        help="profile length in seconds")
    parser.add_argument("--offset", type=int, default=0,
                        help="offset from beginning of the video in seconds")
    parser.add_argument("--trace_path", type=str,
                        help="trace path contains all traces")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # path = args.path
    video_name = args.video
    output_file = args.output
    log_file = args.log
    metadata_file = args.metadata
    short_video_length = args.short_video_length
    profile_length = args.profile_length
    offset = args.offset

    f_trace = open(args.trace_path+'/{}_trace.csv'.format(video_name), 'w')
    f_trace.write('frame id,timestamp,trigger\n')
    tstamp = 0

    det_file = '/data/zxxia/benchmarking/results/videos/{}/720p/profile/updated_gt_FasterRCNN_COCO_no_filter.csv'.format(
        video_name)
    img_path = '/data/zxxia/videos/{}/720p'.format(video_name)
    video = YoutubeVideo(video_name, '720p', metadata_file, det_file, img_path)

    pipeline = Glimpse(
        PARA1_LIST_DICT[video_name], PARA2_LIST, log_file, )  # mode='tracking'

    # choose the first 3 mins to get the best frame diff thresh
    with open(output_file, 'w', 1) as final_result_f:
        result_writer = csv.writer(final_result_f)
        header = ['video chunk', 'para1', 'para2', 'f1',
                  'frame rate', 'frame diff fps', 'tracking fps']
        result_writer.writerow(header)

        # read ground truth and full model detection result
        # image name, detection result, ground truth

        nb_short_videos = (video.frame_count - offset *
                           video.frame_rate)//(short_video_length*video.frame_rate)

        for i in range(nb_short_videos):
            start = i*short_video_length*video.frame_rate+1+offset*video.frame_rate
            end = (i+1)*short_video_length * \
                video.frame_rate+offset*video.frame_rate

            clip = video_name + '_' + str(i)

            profile_start = start
            profile_end = min(start+profile_length*video.frame_rate-1, end)
            test_start = profile_end + 1
            test_end = end
            print("{} {} start={}, end={}".format(clip, img_path, start, end))
            best_para1, best_para2 = pipeline.profile(clip, video,
                                                      profile_start,
                                                      profile_end)

            print("best_para1 = {}, best_para2={}".format(best_para1,
                                                          best_para2))

            ideal_triggered_frame, f1, trigger_f1, pix_change_obj, \
                pix_change_bg, frame_diff_triggered, tracking_triggered, \
                frames_log = pipeline.evaluate(video, test_start, test_end,
                                               best_para1, best_para2)
            # use the selected parameters for the next 5 mins
            frames_triggered = frame_diff_triggered.union(tracking_triggered)
            final_fps = len(frames_triggered) / (test_end - test_start + 1)
            frame_diff_fps = len(frame_diff_triggered) / \
                (test_end - test_start + 1)
            tracking_fps = len(tracking_triggered) / \
                (test_end - test_start + 1)
            result_writer.writerow(
                [clip, best_para1, best_para2, f1, final_fps, frame_diff_fps,
                 tracking_fps])
            for idx in range(start, end + 1):
                if idx in set(range(profile_start, profile_end+1)).union(frames_triggered):
                    f_trace.write(
                        ','.join([str(idx), str(tstamp), str(1)])+'\n')
                else:
                    f_trace.write(
                        ','.join([str(idx), str(tstamp), str(0)]) + '\n')
                tstamp += 1/video.frame_rate
        f_trace.close()


if __name__ == '__main__':
    main()
