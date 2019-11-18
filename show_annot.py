""" Visualize bounding boxes """
import os
import argparse
# import pdb
import cv2
# import numpy as np
# import matplotlib.pyplot as plt
from utils.model_utils import load_full_model_detection, \
        filter_video_detections, eval_single_image, remove_overlappings
from utils.utils import create_dir, compute_f1
from constants import RESOL_DICT, CAMERA_TYPES, COCOLabels
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
    # flag = False
    for box in boxes:
        [xmin, ymin, xmax, ymax, obj_type, score, obj_id] = box
        area = (ymax-ymin)*(xmax-xmin)/(1280*720)
        # if 0.182 <= area <= 0.186:
        #     flag = True
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        if mobilenet_flag:
            cv2.putText(img, '{:.3f}'.format(area), (xmin, ymax-10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            cv2.putText(img, '{:.3f}'.format(score), (xmin+100, ymax-10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            # cv2.putText(img, str(obj_id), (xmin, ymax+10),
            #             cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            cv2.putText(img, str(obj_type), (xmin, ymax+10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)
        else:
            cv2.putText(img, '{:.3f}'.format(area), (xmin, ymin+20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            cv2.putText(img, '{:.3f}'.format(score), (xmin+100, ymin+20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            # cv2.putText(img, str(obj_id), (xmin, ymin-10),
            #             cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            cv2.putText(img, str(obj_type), (xmin, ymin-10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)
    # return flag


def parse_args():
    """ parse input arguments """
    parser = argparse.ArgumentParser(description="Visualization. If "
                                     "resolution2 is specified, bboxes with "
                                     "resolution2 is scaled to resolution1")
    parser.add_argument("--video", type=str, required=True, help="video name")
    parser.add_argument("--detection_file1", type=str, required=True,
                        help="detection file 1")
    parser.add_argument("--resol1", type=str, help="resolution1")
    parser.add_argument("--image_path", type=str, required=True,
                        help="image path")
    parser.add_argument("--detection_file2", type=str, help="detection file 2")
    parser.add_argument("--resol2", type=str, help="resolution2")
    # parser.add_argument("--output", type=str, help="output result file")
    parser.add_argument("--start_frame", type=int, required=True,
                        help="start frame")
    parser.add_argument("--end_frame", type=int, required=True,
                        help="end frame included")
    parser.add_argument("--visualize", action='store_true',
                        help="visualize img on the fly if specified")
    parser.add_argument("--save", action='store_true',
                        help="save img if specified")
    parser.add_argument("--output_folder", type=str, default='vis',
                        help="output folder of saved img if specifie")
    args = parser.parse_args()
    return args


def load_detections(video, dt_file, resol):
    """ load and filter  """
    dts, nb_frame = load_full_model_detection(dt_file)
    if video in CAMERA_TYPES['moving']:
        dts = filter_video_detections(dts,
                                      target_types={COCOLabels.CAR.value,
                                                    COCOLabels.BUS.value,
                                                    COCOLabels.TRAIN.value,
                                                    COCOLabels.TRUCK.value},
                                      height_range=(RESOL_DICT[resol][1]//20,
                                                    RESOL_DICT[resol][1]))
    else:
        dts = filter_video_detections(dts,
                                      target_types={COCOLabels.CAR.value,
                                                    COCOLabels.BUS.value,
                                                    COCOLabels.TRAIN.value,
                                                    COCOLabels.TRUCK.value},
                                      width_range=(0, RESOL_DICT[resol][0]/2),
                                      height_range=(RESOL_DICT[resol][0]//20,
                                                    RESOL_DICT[resol][0]/2))
    for frame_idx, bboxes in dts.items():
        for box_pos, box in enumerate(bboxes):
            box[4] = COCOLabels.CAR.value
            bboxes[box_pos] = box
        dts[frame_idx] = bboxes
    # road_trip
    # for frame_idx in frcnn_dt2:
    #     tmp_boxes = []
    #     for box in frcnn_dt2[frame_idx]:
    #         xmin, ymin, xmax, ymax = box[:4]
    #         if ymin >= 500 and ymax >= 500 and (xmax - xmin) >= 1280/2:
    #             continue
    #         tmp_boxes.append(box)
    #     frcnn_dt2[frame_idx] = tmp_boxes
    for i, boxes in dts.items():
        # import pdb
        # pdb.set_trace()
        dts[i] = remove_overlappings(boxes)

    return dts, nb_frame


def main():
    """ visualize detections """
    args = parse_args()
    dt1, frcnn_frame_cnt1 = \
        load_detections(args.video, args.detection_file1, args.resol1)
    if args.detection_file2 and args.resol2:
        dt2, _ = load_detections(args.video, args.detection_file2, args.resol2)

    assert frcnn_frame_cnt1 >= args.end_frame, \
        "end frame {} > largest frame index {}" \
        .format(args.end_frame, frcnn_frame_cnt1)

    fpos = {}
    fneg = {}
    tpos = {}
    for i in range(args.start_frame, args.end_frame+1):
        print('drawing ', args.video, i)
        img_name = format(i, '06d') + '.jpg'
        img = cv2. imread(args.image_path + img_name)
        visualize(img, dt1[i], (0, 255, 0))
        cv2.putText(img, args.resol1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        if args.resol2 and args.detection_file2:
            boxes = scale_boxes(dt2[i],
                                RESOL_DICT[args.resol2],
                                RESOL_DICT[args.resol1])
            dt2[i] = boxes
            tpos[i], fpos[i], fneg[i] = eval_single_image(dt2[i], dt1[i])
            print('frame {}: tp={}, fp={}, fn={}'.format(i, tpos[i],
                                                         fpos[i], fneg[i]))
            visualize(img, boxes, (255, 0, 0), True)
        if args.visualize:
            cv2.putText(img, '360p', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)
            cv2.imshow(img_name, img)
            cv2.moveWindow(img_name, 200, 200)
            if i > 10:
                cv2.destroyWindow(format(i-10, '06d') + '.jpg')
            if cv2.waitKey(0) & 0xFF == ord('q'):
                args.visualize = False
                cv2.destroyAllWindows()
                # break
        if args.save:
            create_dir(args.output_folder)
            cv2.imwrite(os.path.join(args.output_folder, img_name), img)
            print('write ', args.video, i)
    if args.resol2 and args.detection_file2:
        tp_tot = sum(tpos.values())
        fp_tot = sum(fpos.values())
        fn_tot = sum(fneg.values())
        f1_score = compute_f1(tp_tot, fp_tot, fn_tot)
        print(tp_tot, fp_tot, fn_tot, f1_score)


if __name__ == '__main__':
    main()
