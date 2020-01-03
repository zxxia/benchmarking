from collections import defaultdict
import numpy as np
from benchmarking.utils import IoU


def compute_velocity(start, end, video_dets, fps, step=0.1):
    """No distance info from image, so define a metric to capture the velocity.
    velocity = 1/IoU(current_box, next_box)."""
    velocity = {}

    for i in range(start, end + 1):
        future_frame_idx = int(i + step*fps)
        if future_frame_idx not in video_dets:
            velocity[i] = 0
        else:
            future_obj_map = object_id_to_boxes(video_dets[future_frame_idx])
            boxes = video_dets[i]
            all_velo = []
            for box in boxes:
                obj_id = box[-1]
                # check if this object is also in the frame of 0.1 second ago
                # key = (last_filename, object_id)
                if obj_id not in future_obj_map:
                    continue
                else:
                    # use IoU of the bounding boxes of this object in two
                    # consecutive frames to reflect how fast the object is
                    # moving. the faster the object is moving, the smaller the
                    # IoU will be
                    iou = IoU(future_obj_map[obj_id], box[0:4])
                    if iou < 0.001:
                        print('object {} iou={} too fast from frame {} to frame {}'
                              .format(obj_id, iou, i, future_frame_idx))
                        iou = 0.01
                    # remove the influence of frame rate
                    all_velo.append(1/iou)
                    # all_velo.append(iou)
            velocity[i] = all_velo

    return velocity


def object_id_to_boxes(boxes):
    results = {}
    for box in boxes:
        results[box[-1]] = box
    return results


def object_appearance(start, end, gt):
    """Retun object life span and frames' new object information.

    Return
        object to frame range (dict)
        frame id to a list of new object id (dict)

    """
    obj_to_frame_range = dict()
    frame_to_new_obj = dict()
    for frame_id in range(start, end+1):
        if frame_id not in gt:
            continue
        boxes = gt[frame_id]
        for box in boxes:
            try:
                obj_id = int(box[-1])
            except ValueError:
                obj_id = box[-1]

            if obj_id in obj_to_frame_range:
                start, end = obj_to_frame_range[obj_id]
                obj_to_frame_range[obj_id][0] = min(int(frame_id), start)
                obj_to_frame_range[obj_id][1] = max(int(frame_id), end)
            else:
                obj_to_frame_range[obj_id] = [int(frame_id), int(frame_id)]

    for obj_id in obj_to_frame_range:
        if obj_to_frame_range[obj_id][0] in frame_to_new_obj:
            frame_to_new_obj[obj_to_frame_range[obj_id][0]].append(obj_id)
        else:
            frame_to_new_obj[obj_to_frame_range[obj_id][0]] = [obj_id]

    return obj_to_frame_range, frame_to_new_obj


def compute_arrival_rate(start, end, video_dets, fps):
    """Compuate new object arrival rate.

    Arrival rate defines as the number of new objects arrive in the following
    second starting from the current frame.

    Args
        fps: frames per second

    """

    obj_to_frame_range, frame_to_new_obj = object_appearance(
        start, end, video_dets)
    arrival_rate = {}
    for i in range(start, end + 1 - fps):
        one_second_events = 0
        for frame in range(i, i + fps):
            if frame not in frame_to_new_obj:
                continue
            else:
                one_second_events += len(frame_to_new_obj[i])
        arrival_rate[i] = one_second_events

    return arrival_rate


def compute_nb_object_per_frame(video_detections, start, end):
    """Compute number of object per frame.

    Args
        video_dets(dict): video detections mapping frame index to a list bboxes
        start(int): start frame index
        end(int): end frame index
        resolution(tuple): (width, height)

    Return
        nb_object(dict): mapping frame index to number of object

    """
    nb_object = {}
    for i in range(start, end + 1):
        if i not in video_detections:
            nb_object[i] = 0
        else:
            nb_object[i] = len(video_detections[i])
    return nb_object


def compute_box_size(box):
    """Compute the absolute area of a box in number of pixels."""
    return (box[2]-box[0]+1) * (box[3]-box[1]+1)


def compute_video_object_size(video_dets, start, end, resolution):
    """Compute the size of each object detected in a video.

    Args
        video_dets(dict): video detections mapping frame index to a list bboxes
        start(int): start frame index
        end(int): end frame index
        resolution(tuple): (width, height)

    Return
        object_size(dict): mapping frame index to a list of object size
        total_object_size(dict): mapping frame index to total object size

    """
    object_size = {}
    total_object_size = {}
    for i in range(start, end+1):
        if i not in video_dets:
            continue
        frame_detections = video_dets[i]
        object_size[i] = compute_frame_object_size(
            frame_detections, resolution)
        total_object_size[i] = np.sum(object_size[i])

    return object_size, total_object_size


def compute_frame_object_size(frame_detections, resolution):
    """Compute the size of each object detected in a frame.

    Args
        frame_dets(list): detected bboxes
        resolution(tuple): (width, height)

    Return
        object_size(list): a list of object size

    """
    object_sizes = list()
    image_size = resolution[0]*resolution[1]
    for box in frame_detections:
        object_size = compute_box_size(box)/image_size
        object_sizes.append(object_size)
    return object_sizes
