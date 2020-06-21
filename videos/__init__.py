import glob
import os
from videos.kitti import KittiVideo
from videos.mot15 import MOT15Video
from videos.mot16 import MOT16Video
from videos.waymo import WaymoVideo
from videos.youtube import YoutubeVideo


def get_dataset_class(dataset_name):
    """Return the class template with respect to dataset name."""
    if dataset_name == 'kitti':
        return KittiVideo
    elif dataset_name == 'mot15':
        return MOT15Video
    elif dataset_name == 'mot16':
        return MOT16Video
    elif dataset_name == 'waymo':
        return WaymoVideo
    elif dataset_name == 'youtube':
        return YoutubeVideo
    else:
        raise NotImplementedError


def get_seg_paths(data_path, dataset_name, video_name):
    """Return all segment paths in a dataset."""
    if video_name is not None and video_name:
        seg_paths = [os.path.join(data_path, video_name)]
    elif dataset_name == 'kitti':
        seg_paths = []
        for loc in KittiVideo.LOCATIONS:
            for seg_path in sorted(
                    glob.glob(os.path.join(data_path, loc, '*'))):
                if not os.path.isdir(seg_path):
                    continue
                seg_paths.append(seg_path)
    elif dataset_name == 'mot15':
        seg_paths = []
        for folder in ['test', 'train']:
            for seg_path in sorted(
                    glob.glob(os.path.join(data_path, folder, '*'))):
                if not os.path.isdir(seg_path):
                    continue
                seg_paths.append(seg_path)
    elif dataset_name == 'mot16':
        seg_paths = []
        for folder in ['test', 'train']:
            for seg_path in sorted(
                    glob.glob(os.path.join(data_path, folder, '*'))):
                if not os.path.isdir(seg_path):
                    continue
                seg_paths.append(seg_path)
    elif dataset_name == 'waymo':
        seg_paths = glob.glob(os.path.join(data_path, '*'))
    elif dataset_name == 'youtube':
        seg_paths = glob.glob(os.path.join(data_path, '*'))
    else:
        raise NotImplementedError
    return seg_paths
