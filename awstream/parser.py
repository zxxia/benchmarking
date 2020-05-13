"""AWStream Simulation Parser."""
import argparse


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="AWStream offline simulation.")
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

    # Pipeline configurations
    parser.add_argument("--short_video_length", type=int, required=True,
                        help="Short video length in unit of second.")
    parser.add_argument("--profile_length", type=int, required=True,
                        help="Profile length in unit of second.")
    parser.add_argument("--overfitting", action='store_true',
                        help="Issue when overfitting mode is needed. Only"
                        " spatial pruning (resolution tuning) is kept. "
                        "Temporal pruning (frame rate tuning) and quality "
                        "parameter tuning are dropped.")

    parser.add_argument("--sample_step_list", nargs="*", type=float,
                        default=[20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5,
                                 1.2, 1], help="A list of sample steps. A "
                        "frame is sampled every sample_step frames. This is"
                        "used to change frame rate. Default "
                        "[20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1].")
    parser.add_argument("--original_resolution", type=str,
                        default='720p', help="The resolution used to "
                        "generate groundtruth.")
    parser.add_argument("--resolution_list", nargs="*", type=str,
                        default=['720p', '540p', '480p', '360p'], help="A "
                        "list of resolutions. 720p = (1280, 720). The aspect "
                        "ratio is 16:9 by default. Default "
                        "['720p', '540p', '480p', '360p']")
    parser.add_argument("--quality_parameter_list", nargs="*", type=int,
                        default=[23], help="A list of quality parameters. "
                        "More options are not included for now. In ffmpeg, "
                        "23 by default.")
    parser.add_argument("--classes_interested", nargs="*", type=str,
                        default=['car', 'bus', 'truck'], help="A list of "
                        "interesting classes. Other classes will be filtered "
                        "out. Default ['car', 'bus', 'truck'].")
    parser.add_argument("--coco_label_file", type=str,
                        default='mscoco_label_map.pbtxt', help="Path to a coco"
                        "label map file.")

    # IO realted
    parser.add_argument("--profile_filename", type=str, required=True,
                        help="profile filename (csv). e.g. profile.csv")
    parser.add_argument("--output_filename", type=str, required=True,
                        help="output filename (csv). e.g. output.csv")
    parser.add_argument("--video_save_path", type=str, required=True,
                        help="Video save path where encoded videos will "
                        "be saved to.")
    args = parser.parse_args()
    return args
