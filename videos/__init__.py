from videos.kitti import KittiVideo
from videos.mot15 import MOT15Video
from videos.mot16 import MOT16Video
from videos.video import Video
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
