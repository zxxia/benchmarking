""" Visualize bounding boxes """
# import os
import argparse
# import pdb
import cv2
# import numpy as np
# import matplotlib.pyplot as plt
from utils.model_utils import load_full_model_detection, \
        filter_video_detections  # , eval_single_image
from utils.utils import create_dir  # compute_f1,IoU,
from constants import RESOL_DICT
# from Vigil.vigil_overfitting import load_haar_detection


def scale(box, in_resol, out_resol):
    """
    box: [x, y, w, h]
    in_resl: (width, height)
    out_resl: (width, height)
    """
    assert len(box) >= 4
    ret_box = box.copy()
    x_scale = out_resol[0]/in_resol[0]
    y_scale = out_resol[1]/in_resol[1]
    ret_box[0] = int(box[0] * x_scale)
    ret_box[1] = int(box[1] * y_scale)
    ret_box[2] = int(box[2] * x_scale)
    ret_box[3] = int(box[3] * y_scale)
    return ret_box


def scale_boxes(boxes, in_resol, out_resol):
    """ scale a list of boxes """
    return [scale(box, in_resol, out_resol) for box in boxes]


def visualize(img, boxes, color=(0, 0, 0), mobilenet_flag=False):
    """ visulize """
    for box in boxes:
        if len(box) == 4:
            [xmin, ymin, xmax, ymax] = box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)
            # cv2.putText(img, str(t), (x-10, y-10),
            #             cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            # if mobilenet_flag:
            #     cv2.putText(img, '{:.3f}'.format(score), (x+80, ymax+10),
            #                 cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            #     cv2.putText(img, str(obj_id), (x, ymax+10),
            #                 cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            # else:
            #     cv2.putText(img, '{:.3f}'.format(score), (x+80, y-10),
            #                 cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            #     cv2.putText(img, str(obj_id), (x, y-10),
            #                 cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        else:
            [xmin, ymin, xmax, ymax, obj_type, score, obj_id] = box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)
            if mobilenet_flag:
                cv2.putText(img, '{:.3f}'.format(score), (xmin+80, ymax+10),
                            cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                cv2.putText(img, str(obj_id), (xmin, ymax+10),
                            cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                cv2.putText(img, str(obj_type), (xmin, ymax+10),
                            cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            else:
                cv2.putText(img, '{:.3f}'.format(score), (xmin+80, ymin-10),
                            cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                cv2.putText(img, str(obj_id), (xmin, ymin-10),
                            cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                cv2.putText(img, str(obj_type), (xmin, ymin-10),
                            cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)


def main():
    """ visualize detections """
    parser = argparse.ArgumentParser(description="Visualization. If "
                                     "resolution2 is specified, bboxes with "
                                     "resolution2 is scaled to resolution1")
    parser.add_argument("--video", type=str, required=True, help="video name")
    parser.add_argument("--detection_file1", type=str, required=True,
                        help="detection file 1")
    parser.add_argument("--image_path1", type=str, required=True,
                        help="image path 1")
    parser.add_argument("--detection_file2", type=str, help="detection file 2")
    parser.add_argument("--image_path2", type=str, help="image path 2")
    # parser.add_argument("--output", type=str, help="output result file")
    parser.add_argument("--start_frame", type=int, required=True,
                        help="start frame")
    parser.add_argument("--end_frame", type=int, required=True,
                        help="end frame included")
    parser.add_argument("--visualize", action='store_true',
                        help="visualize img on the fly if specified")
    parser.add_argument("--save", action='store_true',
                        help="save img if specified")

    args = parser.parse_args()
    visualize_flag = args.visualize
    frcnn_file2 = args.detection_file2

    frcnn_dt1, frcnn_frame_cnt1 = \
        load_full_model_detection(args.detection_file1)
    # frcnn_dt1 = filter_video_detections(frcnn_dt1,
    #                                     width_range=(0, 1280/2),
    #                                     height_range=(720//20, 720/2),
    #                                     target_types={3, 8})
    frcnn_dt1 = filter_video_detections(frcnn_dt1,
                                        height_range=(720//20, 720),
                                        target_types={3, 8})
    # for frame_idx, bboxes in frcnn_dt1.items():
    #     for box_pos, box in enumerate(bboxes):
    #         box[4] = 3
    #         bboxes[box_pos] = box
    #     frcnn_dt1[frame_idx] = bboxes
    # road_trip
    for frame_idx in frcnn_dt1:
        tmp_boxes = []
        for box in frcnn_dt1[frame_idx]:
            xmin, ymin, xmax, ymax = box[:4]
            if ymin >= 500 and ymax >= 500 and (xmax - xmin) >= 1280/2:
                continue
            tmp_boxes.append(box)
        frcnn_dt1[frame_idx] = tmp_boxes
    if frcnn_file2 and args.image_path2:
        frcnn_dt2, _ = load_full_model_detection(frcnn_file2)
        # frcnn_dt2 = filter_video_detections(frcnn_dt2,
        #                                     width_range=(0, 1280/2),
        #                                     height_range=(720//20, 720/2),
        #                                     target_types={3, 8},
        #                                     score_range=(0.4, 1.0))
        frcnn_dt2 = filter_video_detections(frcnn_dt2,
                                            height_range=(720//20, 720),
                                            target_types={3, 8})
        # for frame_idx, bboxes in frcnn_dt2.items():
        #     for box_pos, box in enumerate(bboxes):
        #         box[4] = 3
        #         bboxes[box_pos] = box
        #     frcnn_dt2[frame_idx] = bboxes

        # road_trip
        for frame_idx in frcnn_dt2:
            tmp_boxes = []
            for box in frcnn_dt2[frame_idx]:
                xmin, ymin, xmax, ymax = box[:4]
                if ymin >= 500 and ymax >= 500 and (xmax - xmin) >= 1280/2:
                    continue
                tmp_boxes.append(box)
            frcnn_dt2[frame_idx] = tmp_boxes

    assert frcnn_frame_cnt1 >= args.end_frame, \
        "end frame {} > largest frame index {}" \
        .format(args.end_frame, frcnn_frame_cnt1)

    # fpos = {}
    # fneg = {}
    # tpos = {}
    for i in range(args.start_frame, args.end_frame+1):
        print('drawing ', args.video, i)
        img_name = format(i, '06d') + '.jpg'
        img1 = cv2. imread(args.image_path1 + img_name)
        visualize(img1, frcnn_dt1[i], (0, 255, 0))
        if args.image_path2 and frcnn_file2:
            img2 = cv2. imread(args.image_path2 + img_name)
            boxes = scale_boxes(frcnn_dt2[i],
                                RESOL_DICT[str(img2.shape[0])+'p'],
                                RESOL_DICT[str(img1.shape[0])+'p'])
            visualize(img1, boxes, (255, 0, 0))
        if visualize_flag:
            cv2.imshow(img_name, img1)
            cv2.moveWindow(img_name, 200, 200)
            if i > 10:
                cv2.destroyWindow(format(i-10, '06d') + '.jpg')
            if cv2.waitKey(0) & 0xFF == ord('q'):
                visualize_flag = False
                cv2.destroyAllWindows()
                # break
        if args.save:
            output_path = "vis/"
            create_dir(output_path)
            cv2.imwrite(output_path + img_name, img1)
            print('write ', args.video, i)
    #     tpos[i], fpos[i], fneg[i] = eval_single_image(frcnn_dt2[i],
    #                                                   frcnn_dt1[i])
    #     print('frame {}: tp={}, fp={}, fn={}'.format(i, tpos[i],
    #                                                  fpos[i], fneg[i]))
    # tp_tot = sum(tpos.values())
    # fp_tot = sum(fpos.values())
    # fn_tot = sum(fneg.values())
    # f1_score = compute_f1(tp_tot, fp_tot, fn_tot)
    # print(tp_tot, fp_tot, fn_tot, f1_score)


if __name__ == '__main__':
    main()

# def match_boxes(mb_boxes, frcnn_boxes):
#     m_scores = []
#     f_scores = []
#     for m_box in mb_boxes:
#         ious = []
#         for f_box in frcnn_boxes:
#             iou = IoU(m_box, f_box)
#             ious.append(iou)
#         if max(ious) > 0.5:
#             max_idx = np.argmax(ious)
#             # print(max_idx)
#             m_scores.append(m_box[5])
#             f_scores.append(frcnn_boxes[max_idx][5])
#     return m_scores, f_scores
#
#
# def obj_to_frames(gt):
#     results = defaultdict(list)
#     for frame_id in sorted(gt.keys()):
#         for box in gt[frame_id]:
#             obj_id = box[6]
#             results[obj_id].append(frame_id)
#     return results
#
#
# def match_obj_ids(boxes_A, boxes_B, id_to_ids):
#     for box_A in boxes_A:
#         ious = []
#         for box_B in boxes_B:
#             iou = IoU(box_A, box_B)
#             ious.append(iou)
#         if max(ious) > 0.8:
#             max_idx = np.argmax(ious)
#             # print(max_idx)
#             id_to_ids[box_A[6]].add(boxes_B[max_idx][6])
#     return id_to_ids


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

# def plot_cdf(data, num_bins, title, legend, xlabel, xlim=None):
#     # Use the histogram function to bin the data
#     counts, bin_edges = np.histogram(data, bins=num_bins, density=True)
#     dx = bin_edges[1] - bin_edges[0]
#     # Now find the cdf
#     cdf = np.cumsum(counts) * dx
#     # And finally plot the cdf
#     plt.plot(bin_edges[1:], cdf, label=legend)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel('CDF')
#     plt.ylim([0, 1.1])
#     if xlim is not None:
#         plt.xlim(xlim)
#     plt.legend()
# haar_dets = load_haar_detection(
# '/data/zxxia/benchmarking/Vigil/haar_detections_new/haar_{}.csv'
# .format(video_name))
