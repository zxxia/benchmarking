"""Functions to compute object-level features."""
import numpy as np
from evaluation.f1 import IoU


def compute_velocity(video_dets, start, end, fps, step=0.1, sample_step=1):
    """Compute object velocity.

    No distance info from image, so define a metric to capture the velocity.
    velocity = 1/IoU(current_box, next_box).

    Args
        video_dets(dict): a dict mapping frame index to a list of bboxes
        start(int): start frame index
        end(int): end frame index
        step(float): the time step which are used to compute velocity
        sample_step(int): sample a frame every sample_step frames

    Return
        velocity(dict): a dict mapping frame index to a list of object
                        velocities

    """
    velocity = {}

    for i in range(start, end + 1, sample_step):
        boxes = video_dets[i]
        past_frame_idx = int(i - step*fps)
        if past_frame_idx not in video_dets:
            velocity[i] = []
            # if i - 1 in velocity:
            #     # TODO: may need to check objec id existence here
            #     velocity[i] = velocity[i-1]
        else:
            past_obj_map = object_id_to_boxes(video_dets[past_frame_idx])
            all_velo = []
            for box in boxes:
                obj_id = box[-1]
                # check if this object is also in the frame of 0.1 second ago
                # key = (last_filename, object_id)
                if obj_id not in past_obj_map:
                    continue
                else:
                    # use IoU of the bounding boxes of this object in two
                    # consecutive frames to reflect how fast the object is
                    # moving. the faster the object is moving, the smaller the
                    # IoU will be
                    iou = IoU(box[0:4], past_obj_map[obj_id])
                    if iou < 0.1:
                        # print('object {} iou={} too fast from frame {} to frame {}'
                        #       .format(obj_id, iou, i, past_frame_idx))
                        iou = 0.1
                    # remove the influence of frame rate
                    all_velo.append(1/iou)
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


def compute_arrival_rate(video_dets, start, end, fps):
    """Compuate new object arrival rate.

    Arrival rate defines as the number of new objects arrive in the following
    second starting from the current frame.

    Args
        fps: frames per second

    """
    obj_to_frame_range, frame_to_new_obj = object_appearance(
        start, end, video_dets)
    arrival_rate = {}
    for i in range(start, end + 1):
        one_second_events = 0
        if i > end - fps:
            arrival_rate[i] = arrival_rate[i - 1]
        for j in range(i, i + fps):
            if j not in frame_to_new_obj:
                continue
            else:
                one_second_events += len(frame_to_new_obj[j])
        arrival_rate[i] = one_second_events

    return arrival_rate


def compute_nb_object_per_frame(video_detections, start, end, sample_step=1):
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
    for i in range(start, end + 1, sample_step):
        if i not in video_detections:
            nb_object[i] = 0
        else:
            nb_object[i] = len(video_detections[i])

    return nb_object


def compute_box_size(box):
    """Compute the absolute area of a box in number of pixels."""
    return (box[2]-box[0]+1) * (box[3]-box[1]+1)


def compute_video_object_size(video_dets, start, end, resolution,
                              sample_step=1):
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
    for i in range(start, end+1, sample_step):
        if i not in video_dets:
            continue
        frame_detections = video_dets[i]
        object_size[i] = compute_frame_object_size(
            frame_detections, resolution)
        total_object_size[i] = compute_frame_total_object_size(
            frame_detections, resolution)

    return object_size, total_object_size


def compute_frame_total_object_size(frame_detections, resolution):
    """Compute the size of all objects detected in a frame.

    Args
        frame_dets(list): detected bboxes
        resolution(tuple): (width, height)

    Return
        total_object_size(int):

    """
    image_size = resolution[0]*resolution[1]
    pixels = np.zeros(resolution)
    for box in frame_detections:
        xmin, ymin, xmax, ymax = box[:4]
        pixels[int(xmin):int(xmax), int(ymin):int(ymax)] = 1
    return np.sum(pixels) / image_size


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


def count_unique_class(video_dets, start, end, sample_step=1):
    """Count unique classes in a video.

    Args
        video_dets(dict): a dict mapping frame index to a list of bboxes

    Return
        unique class count

    """
    unique_classes = set()
    for i in range(start, end+1, sample_step):
        if i not in video_dets:
            continue
        for box in video_dets[i]:
            unique_classes.add(box[4])
    return len(unique_classes)


def count_classification_unique_class(video_dets, start, end, sample_step=1):
    """Count unique classes in a video using classification.

    Args
        video_dets(dict): a dict mapping frame index to a list of bboxes

    Return
        unique class count

    """
    unique_classes = set()
    for i in range(start, end+1, sample_step):
        if i not in video_dets:
            continue
        for box in video_dets[i]:
            unique_classes.add(box[4])
        # print(unique_classes)
    return len(unique_classes)


def compute_percentage_frame_with_object(video_dets, start, end,
                                         sample_step=1):
    """Compute the percentage of frames with object in a video.

    Args
        video_dets(dict): a dict mapping frame index to a list of bboxes
        start(int): start frame
        end(int): end frame
        sample_step(int): sample every sample_step steps

    Return
        percentage

    """
    cnt = 0
    for i in range(start, end+1, sample_step):
        if video_dets[i]:
            cnt += 1
    return cnt / (end-start+1)


def compute_percentage_frame_with_new_object(video_dets, start, end):
    """Compute the percentage of frames with new object in a video.

    Args
        video_dets(dict): a dict mapping frame index to a list of bboxes
        start(int): start frame
        end(int): end frame

    Return
        percentage

    """
    object_first_frame = {}
    for i in range(start, end+1):
        boxes = video_dets[i]
        for box in boxes:
            _id = box[6]
            # if this object not exist before, this frame is its first frame
            if _id not in object_first_frame:
                object_first_frame[_id] = i

    return len(set(object_first_frame.values())) / (end-start+1)
