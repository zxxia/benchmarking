"""performance_estimator profiler Parser."""
import argparse


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Pipeline profiler.")
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

    parser.add_argument("--sample_step_list", nargs="*", type=float,
                        default=[20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5,
                                 1.2, 1], help="A list of sample steps. A "
                        "frame is sampled every sample_step frames. This is"
                        "used to change frame rate. Default "
                        "[20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1].")
    parser.add_argument("--original_resolution", type=str,
                        default='720p', help="The resolution used to "
                        "generate groundtruth.")
    parser.add_argument("--model_list", nargs="*", type=str,
                        default=['FasterRCNN', 'inception', 'mobilenet'],
                        help="A list of quality parameters. "
                        "More options are not included for now. In ffmpeg, "
                        "23 by default.")
    parser.add_argument("--classes_interested", nargs="*", type=str,
                        default=['car', 'bus', 'truck'], help="A list of "
                        "interesting classes. Other classes will be filtered "
                        "out. Default ['car', 'bus', 'truck'].")
    parser.add_argument("--coco_label_file", type=str,
                        default='mscoco_label_map.pbtxt', help="Path to a coco"
                        "label map file.")
    # feature scanner granularity
    parser.add_argument("--granularity", type=str, required=True,
                        choices=['high', 'med', 'low'],
                        help="granularity of feature scan")
    #profiler related
    parser.add_argument("--interested_feature_list", nargs="*", type=str,
                        default=['percent_of_frame_w_object', 'velocity avg','object area percentile10','total area avg', 'avg confidence score'], help="A list of "
                        "interested feature. Other features will be filtered "
                        "out. Default ['percent_of_frame_w_object', 'velocity avg','object area percentile10','total area avg', 'avg confidence score'].")
    parser.add_argument("--interested_performance_list", nargs="*", type=str,
                        default=['f1', 'gpu time'], help="A list of "
                        "interested performance list. Other performance will be filtered "
                        "out. Default ['f1', 'gpu time'].")
    parser.add_argument("--temporal_feature_list", nargs="*", type=str,
                        default=['percent_of_frame_w_object', 'velocity avg','avg confidence score' ], help="A list of "
                        "features that will affect the temporal pruning performance. "
                        ". Default ['percent_of_frame_w_object', 'velocity avg','avg confidence score'].")                  
    parser.add_argument("--spatial_feature_list", nargs="*", type=str,
                        default=['object area percentile10', 'percent_of_frame_w_object','total area avg'], help="A list of "
                        "features that will affect the spatial pruning performance. "
                        ". Default ['object area percentile10', 'percent_of_frame_w_object','total area avg'].")
    parser.add_argument("--model_feature_list", nargs="*", type=str,
                        default=['object area percentile10', 'avg confidence score'], help="A list of "
                        "features that will affect the model pruning performance. "
                        ". Default ['object area percentile10', 'avg confidence score'].")
    parser.add_argument("--interested_feature_bucket_info_dict",  type=str,
                        default="{'percent_of_frame_w_object': (0.9, 1.0, 3), 'velocity avg': (1.6, 3.0, 3),'object area "
                        "percentile10':(0, 0.005, 3),'total area avg':(0.2, 0.9, 3), 'avg confidence score':(0.7, 1, 3)}",
                        help="A string that will be pared by eval() to get a dict, the key should be feature name, the value should"
                        "be a tuple of (min, max, num_of_bucket). Note that the sequence should have the same features and order as the "
                        "interested_feature_list argument!")
    # IO realted
    parser.add_argument("--feature_output_filename", type=str, default='feature_output.csv',
                        help="feature output filename (csv). e.g. feature_output.csv")
    parser.add_argument("--pipeline_performance_filename", type=str, default='pipeline_performance_output.csv',
                        help="feature output filename (csv). e.g. pipeline_performance_output.csv")
    parser.add_argument("--temporal_pruning_profile_filename", type=str, default='temporal_pruning_profile_output.json',
                        help="temporal_pruning_profile_filename (json). e.g. temporal_pruning_profile_output.json")
    parser.add_argument("--spatial_pruning_profile_filename", type=str, default='spatial_pruning_profile_output.json',
                        help="spatial_pruning_profile_filename (json). e.g. spatial_pruning_profile_output.json")
    parser.add_argument("--model_pruning_profile_filename", type=str, default='model_pruning_profile_output.json',
                        help="model_pruning_profile_filename (json). e.g. model_pruning_profile_output.json")
    
    args = parser.parse_args()
    return args
