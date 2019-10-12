import os
import cv2
# from absl import app, flags
from collections import defaultdict
from utils.model_utils import load_full_model_detection
from utils.utils import IoU
import numpy as np
import matplotlib.pyplot as plt
import pdb
import argparse
from utils.utils import create_dir


IMAGE_RESOLUTION_DICT = {'360p': [640, 360],
                         '480p': [854, 480],
                         '540p': [960, 540],
                         '576p': [1024, 576],
                         '720p': [1280, 720],
                         '1080p': [1920, 1080],
                         '2160p': [3840, 2160]}


def scale(box, in_resol, out_resol):
    """
    box: [x, y, w, h]
    in_resl: (width, height)
    out_resl: (width, height)
    """
    assert(len(box) >= 4)
    ret_box = box.copy()
    x_scale = out_resol[0]/in_resol[0]
    y_scale = out_resol[1]/in_resol[1]
    ret_box[0] = int(box[0] * x_scale)
    ret_box[1] = int(box[1] * y_scale)
    ret_box[2] = int(box[2] * x_scale)
    ret_box[3] = int(box[3] * y_scale)
    return ret_box


def plot_cdf(data, num_bins, title, legend, xlabel, xlim=None):
    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=num_bins, density=True)
    dx = bin_edges[1] - bin_edges[0]
    # Now find the cdf
    cdf = np.cumsum(counts) * dx
    # And finally plot the cdf
    plt.plot(bin_edges[1:], cdf, label=legend)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.ylim([0, 1.1])
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()


def visualize(img, frame_id, boxes, color=(0, 0, 0), mobilenet_flag=False):
    for box in boxes:
        [x, y, xmax, ymax, t, score, obj_id] = box
        cv2.rectangle(img, (x, y), (xmax, ymax), color, 3)
        # cv2.putText(img, str(t), (x-10, y-10),
        #             cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        if mobilenet_flag:
            cv2.putText(img, '{:.3f}'.format(score), (x+80, ymax+10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            cv2.putText(img, str(obj_id), (x, ymax+10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        else:
            cv2.putText(img, '{:.3f}'.format(score), (x+80, y-10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            cv2.putText(img, str(obj_id), (x, y-10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)


# def get_frame_to_object(gt_file):
#     frame_to_object = defaultdict(list)
#     with open(gt_file, 'r') as f:
#         for line in f:
#             line_list = line.strip().split(',')
#             frame_id = int(line_list[0])
#             # frame_cn = int(line_list[0])
#             # object_id = line_list[2]2
#             if line_list[1] == '':
#                 continue
#             else:
#                 gt_str = line_list[1].split(';')
#                 for box_str in gt_str:
#                     box = box_str.split(' ')
#                     frame_to_object[frame_id].append(box)
#     return frame_to_object

def match_boxes(mb_boxes, frcnn_boxes):
    m_scores = []
    f_scores = []
    for m_box in mb_boxes:
        ious = []
        for f_box in frcnn_boxes:
            iou = IoU(m_box, f_box)
            ious.append(iou)
        if max(ious) > 0.5:
            max_idx = np.argmax(ious)
            # print(max_idx)
            m_scores.append(m_box[5])
            f_scores.append(frcnn_boxes[max_idx][5])
    return m_scores, f_scores


def obj_to_frames(gt):
    results = defaultdict(list)
    for frame_id in sorted(gt.keys()):
        for box in gt[frame_id]:
            obj_id = box[6]
            results[obj_id].append(frame_id)
    return results


def match_obj_ids(boxes_A, boxes_B, id_to_ids):
    for box_A in boxes_A:
        ious = []
        for box_B in boxes_B:
            iou = IoU(box_A, box_B)
            ious.append(iou)
        if max(ious) > 0.8:
            max_idx = np.argmax(ious)
            # print(max_idx)
            id_to_ids[box_A[6]].add(boxes_B[max_idx][6])
    return id_to_ids


def main():
    parser = argparse.ArgumentParser(description="Visualization. If "
                                     "resolution2 is specified, bboxes with "
                                     "resolution2 is scaled to resolution1")
    parser.add_argument("--video", type=str, help="video name")
    parser.add_argument("--output", type=str, help="output result file")
    parser.add_argument("--start_frame", type=int, help="start frame")
    parser.add_argument("--end_frame", type=int, help="end frame included")
    parser.add_argument("--resolution1", type=int, help="video resolution1")
    parser.add_argument("--resolution2", type=int, help="video resolution2")
    parser.add_argument("--visualize", action='store_true',
                        help="visualize img on the fly if specified")

    args = parser.parse_args()
    video_name = args.video
    start_frame = args.start_frame
    end_frame = args.end_frame
    visualiz_flag = args.visualize
    resolution1 = args.resolution1
    resolution2 = args.resolution2

    path = "/mnt/data/zhujun/dataset/Youtube/{}/".format(video_name)
    frcnn_file1 = '{}{}p/profile/updated_gt_FasterRCNN_COCO.csv' \
        .format(path, resolution1)
    frcnn_dt1, frcnn_frame_cnt1 = load_full_model_detection(frcnn_file1,
                                                            # score_range=[0.5, 1],
                                                            height=resolution1)
    img_path = path + '/' + str(resolution1) + 'p/'
    if resolution2:
        frcnn_file2 = '{}{}p/profile/updated_gt_FasterRCNN_COCO.csv' \
                      .format(path, resolution2)
        frcnn_dt2, frcnn_frame_cnt2 = \
            load_full_model_detection(frcnn_file2,
                    # score_range=[0.5, 1],
                    height=resolution2)
        # assert frcnn_frame_cnt1 == frcnn_frame_cnt2, \
        #     'inconsistent frame count {} != {}'.format(frcnn_frame_cnt1,
        # frcnn_frame_cnt2)

    # output_path = "vis/" \
                  # .format(video_name, resolution1)

    assert frcnn_frame_cnt1 >= end_frame, \
        "end frame {} > largest frame index {}" \
        .format(end_frame, frcnn_frame_cnt1)
    for i in range(start_frame, end_frame+1):
        print('drawing ', video_name, i)
        img_name = format(i, '06d') + '.jpg'
        img = cv2. imread(img_path + img_name)
        visualize(img, img_path, frcnn_dt1[i], (0, 255, 0))
        if resolution2:
            boxes = []
            for box in frcnn_dt2[i]:
                boxes.append(scale(box, IMAGE_RESOLUTION_DICT[str(resolution2)+'p'],
                             IMAGE_RESOLUTION_DICT[str(resolution1)+'p']))
            visualize(img, img_path, boxes, (255, 0, 0))
        if visualiz_flag:
            cv2.imshow(img_name, img)
            cv2.moveWindow(img_name, 200, 200)
            if i > 10:
                cv2.destroyWindow(format(i-10, '06d') + '.jpg')
            c = cv2.waitKey(0)
            if c & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            output_path = "vis/"
            create_dir(output_path)
            cv2.imwrite(output_path + img_name, img)
            print('write ', video_name, i)


if __name__ == '__main__':
    main()
