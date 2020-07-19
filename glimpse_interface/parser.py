"""VideoStorm Simulation Parser."""
import argparse


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="VideoStorm offline simulation.")
    # data related
    parser.add_argument("--video", type=str, default=None, help="video name")
    parser.add_argument("--data_root", type=str, required=True,
                        help='root path to video mp4/frame data')
    # parser.add_argument("--detection_root", type=str, required=True,
    #                     help='root path to video detection data')
    parser.add_argument("--dataset", type=str, default='youtube',
                        choices=['kitti', 'mot15',
                                 'mot16', 'waymo', 'youtube'],
                        help="dataset name")
    parser.add_argument("--original_resolution", type=str,
                        default='720p', help="The resolution used to "
                        "generate groundtruth.")
    parser.add_argument("--coco_label_file", type=str,
                        default='mscoco_label_map.pbtxt', help="Path to a coco"
                        "label map file.")

    # Pipeline configurations
    parser.add_argument("--short_video_length", type=int, required=True,
                        help="Short video length in unit of second.")
    parser.add_argument("--profile_length", type=int, required=True,
                        help="Profile length in unit of second.")
    parser.add_argument("--overfitting", action='store_true',
                        help="Issue when overfitting mode is needed. Tuning "
                        "config over the entire short_video_length.")

    parser.add_argument("--frame_difference_threshold_divisor",
                        type=float, default=2,
                        help="frame difference threshold divisor. "
                        "frame difference threshold = width * height / divior."
                        )
    parser.add_argument("--tracking_error_threshold",
                        type=float, default=1,
                        help="tracking error thresholds. Only used"
                        "in optical flow tracking.")
    parser.add_argument("--cache_size",
                        type=int, default=30,
                        help="size of active cache.")
    parser.add_argument("--sample_rate",
                        type=int, default=5,
                        help="the sample rate of frames in active cache.")
    parser.add_argument("--classes_interested", nargs="*", type=str,
                        default=['car', 'bus', 'truck'], help="A list of "
                        "interesting classes. Other classes will be filtered "
                        "out. Default ['car', 'bus', 'truck'].")
    # IO realted
    parser.add_argument("--profile_filename", type=str, required=True,
                        help="profile filename (csv). e.g. profile.csv")
    parser.add_argument("--output_filename", type=str, required=True,
                        help="output filename (csv). e.g. output.csv")
    parser.add_argument("--output_video", type=str, default="output.avi",
                        help="the video name if need to visualize the result")
    parser.add_argument("--vis_video", action='store_true', help="visualize the results in video")
    args = parser.parse_args()
    return args
