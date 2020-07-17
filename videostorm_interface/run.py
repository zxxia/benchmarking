import csv
import os

from utils.utils import load_COCOlabelmap
from videos import get_dataset_class, get_seg_paths
from videostorm_interface.VideoStorm import VideoStorm, VideoStorm_Temporal, VideoStorm_Spacial, VideoStorm_Model

def run(args):
    """Run VideoStorm simulation."""
    dataset_class = get_dataset_class(args.dataset)
    seg_path = get_seg_paths(args.data_root, args.dataset, args.vedio)
    original_resolution = 