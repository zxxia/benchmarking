"""Vigil Implementation."""
import os
from benchmarking.utils.model_utils import eval_single_image
from benchmarking.utils.utils import compute_f1


class Vigil():
    """Vigil Implementation."""

    def __init__(self):
        pass

    def evaluate(self, video_name, video, original_video, frame_range,
                 video_save_path):
        """Evaluate."""
        tpos = {}
        fpos = {}
        fneg = {}
        for i in range(frame_range[0], frame_range[1] + 1):

            # relative uploaded area
            # relative_up_area = 0
            # simple_dt_boxes = mobilenet_dt[img_idx]
            # for simple_dt_box in simple_dt_boxes:
            #     simple_dt_box = resize_bbox(simple_dt_box, 0.2, 0.2,
            #                                 resolution)
            #     xmin, ymin, xmax, ymax = simple_dt_box[:4]
            #     relative_up_area += (xmax-xmin)*(ymax-ymin) /\
            #         (resolution[0]*resolution[1])
            # relative_up_areas.append(relative_up_area)

            # tot_obj_area = 0
            # for gt_box in gt_boxes:
            #     xmin, ymin, xmax, ymax = gt_box[:4]
            #     tot_obj_area += (xmax-xmin)*(ymax-ymin) /\
            #         (resolution[0]*resolution[1])
            # tot_obj_areas.append(tot_obj_area)

            tpos[i], fpos[i], fneg[i] = eval_single_image(
                original_video.get_frame_detection(i),
                video.get_frame_detection(i))
        tp_total = sum(tpos.values())
        fp_total = sum(fpos.values())
        fn_total = sum(fneg.values())
        f1_score = compute_f1(tp_total, fp_total, fn_total)
        # TODO: change video name
        original_bw = original_video.encode(
            os.path.join(video_save_path, video_name +
                         '.mp4'), list(range(frame_range[0], frame_range[1] + 1)),
            original_video.frame_rate)
        bw = video.encode(os.path.join(video_save_path, video_name+'.mp4'),
                          list(range(frame_range[0], frame_range[1] + 1)),
                          video.frame_rate)
        return bw/original_bw, f1_score


def resize_bboxes(bboxes, w_delta_percent, h_delta_percent, resolution):
    """Resize a bounding box.

    Args
        bboxes(list): a list of boxes.
                      [xmin, ymin, xmax, ymax, t, score, obj_id]
        w_delta_percent(float): percent of change over w
        h_delta_percent(float): percent of change over h

    """
    ret_bboxes = [resize_bbox(bbox, w_delta_percent, h_delta_percent,
                              resolution) for bbox in bboxes]
    return ret_bboxes


def resize_bbox(bbox, w_delta_percent, h_delta_percent, resolution):
    """Resize a bounding box.

    box format is [xmin, ymin, xmax, ymax, t, score, obj_id]

    """
    ret_bbox = bbox.copy()
    xmin, ymin, xmax, ymax = bbox[:4]
    w_delta = int(0.5 * (xmax - xmin) * w_delta_percent)
    h_delta = int(0.5 * (ymax - ymin) * h_delta_percent)
    ret_bbox[0] = max(xmin-w_delta, 0)
    ret_bbox[1] = max(ymin-h_delta, 0)
    ret_bbox[2] = min(xmax+w_delta, resolution[0])
    ret_bbox[3] = min(ymax+h_delta, resolution[1])
    return ret_bbox
