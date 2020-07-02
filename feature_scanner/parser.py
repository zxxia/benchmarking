"""Feature Scanner Parser."""
import argparse


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description="Feature Scanner.")
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
    parser.add_argument("--granularity", type=str, required=True,
                        choices=['high', 'med', 'low'],
                        help="granularity of feature scan")

    parser.add_argument("--classes_interested", nargs="*", type=str,
                        default=['car', 'bus', 'truck'], help="A list of "
                        "interesting classes. Other classes will be filtered "
                        "out. Default ['car', 'bus', 'truck'].")
    parser.add_argument("--coco_label_file", type=str,
                        default='mscoco_label_map.pbtxt', help="Path to a coco"
                        "label map file.")

    # IO realted
    parser.add_argument("--output_filename", type=str, required=True,
                        help="output filename (csv). e.g. output.csv")
    args = parser.parse_args()
    return args
