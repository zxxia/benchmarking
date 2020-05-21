"""Vigil Simulation Parser."""
import argparse


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Vigil offline simulation.")
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
    parser.add_argument("--simple_model", type=str, default='ssd_mobilenet_v2',
                        choices=['faster_rcnn_resnet101',
                                 'faster_rcnn_inception_v2',
                                 'ssd_mobilenet_v2'],
                        help="simple model used to crop frames.")
    parser.add_argument("--overfitting", action='store_true',
                        help="Issue when overfitting mode is needed. Only"
                        " spatial pruning (resolution tuning) is kept. "
                        "Temporal pruning (frame rate tuning) and quality "
                        "parameter tuning are dropped.")
    parser.add_argument("--crop", action='store_true',
                        help="Crop frames using the detected bounding boxes.")

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
    parser.add_argument("--output_filename", type=str, required=True,
                        help="output filename (csv). e.g. output.csv")
    parser.add_argument("--video_save_path", type=str, required=True,
                        help="Video save path where encoded videos will "
                        "be saved to.")
    args = parser.parse_args()
    return args
