"""Reducto Simulation Parser."""
import argparse


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description="Reducto simulation.")
    # data related
    parser.add_argument("--video", type=str, default=None, help="video name")
    parser.add_argument("--data_root", type=str, required=True,
                        help='root path to video mp4/frame data')
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
                        " temporal pruning (frame rate tuning) is kept. "
                        "Model selection is kept with faster_rcnn_resenet101.")

    parser.add_argument("--original_resolution", type=str,
                        default='720p', help="The resolution used to "
                        "generate groundtruth.")
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
    parser.add_argument("--save_dir", type=str, default=None,
                        help="directory where encoded videos will be saved to.")
    args = parser.parse_args()
    return args
