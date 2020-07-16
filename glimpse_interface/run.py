"""Glimpse offline simulation driver."""
import csv
import os

from glimpse_interface.Glimpse import Glimpse, Glimpse_Temporal, Glimpse_Model
from videos import get_dataset_class, get_seg_paths
from utils.utils import load_COCOlabelmap


def run(args):
    """Run Glimpse simulation."""
    dataset_class = get_dataset_class(args.dataset)
    seg_paths = get_seg_paths(args.data_root, args.dataset, args.video)
    original_resolution = args.original_resolution
    coco_id2name, coco_name2id = load_COCOlabelmap(args.coco_label_file)
    classes_interested = {coco_name2id[class_type]
                          for class_type in set(args.classes_interested)}
    overfitting = args.overfitting
    tracking_error_threshold_list = args.tracking_error_threshold_list
    frame_difference_divisor_list = \
        args.frame_difference_threshold_divisor_list
    short_video_length = args.short_video_length
    profile_length = args.profile_length
    # detection_path = args.detection_root
    profile_filename = args.profile_filename
    output_filename = args.output_filename

    if overfitting:
        assert short_video_length == profile_length, "short_video_length " \
            "should equal to profile_length when overfitting."
    else:
        assert short_video_length >= profile_length, "short_video_length " \
            "should no less than profile_length."
    glimpse_temporal = Glimpse_Temporal(30, 5, 1280*720//2, 1)
    glimpse_model = Glimpse_Model(1280*720//2, 1)
    pipeline = Glimpse(glimpse_temporal, glimpse_model)
    for seg_path in seg_paths:
        seg_name = os.path.basename(seg_path)
        original_video = dataset_class(
                    seg_path, seg_name, original_resolution,
                    'faster_rcnn_resnet101', filter_flag=True,
                    classes_interested=classes_interested)
        pipeline.run(original_video, original_video.start_frame_index, original_video.end_frame_index, output_filename)
        