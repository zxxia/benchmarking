""" object size vs accuracy """
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from constants import CAMERA_TYPES, COCOLabels, RESOL_DICT
from utils.utils import IoU, compute_f1
from utils.model_utils import load_full_model_detection, \
    filter_video_detections, eval_single_image, remove_overlappings, compute_area
from awstream.profiler import scale_boxes
from feature_analysis.helpers import load_awstream_profile

ROOT = '/data/zxxia/benchmarking/results/videos'
VIDEO_TO_DELETE = ['crossroad', 'nyc', 'russia', 'crossroad2',
                   'driving_downtown',
                   'tw_road', 'tw_under_bridge', 'tw1', 'tw', 'crossroad3']
DATA_PATH = '/data/zxxia/videos'


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
        # merge all vehicle labels into CAR
        for box_pos, box in enumerate(bboxes):
            box[4] = COCOLabels.CAR.value
            bboxes[box_pos] = box
        dts[frame_idx] = bboxes
        # remove overlappings to mitigate occultation
        dts[frame_idx] = remove_overlappings(bboxes, 0.3)
    return dts, nb_frame


def main():
    """ main """
    video_list = sorted(['crossroad', 'crossroad2', 'crossroad3', 'crossroad4',
                         'drift', 'driving1', 'driving_downtown', 'lane_split',
                         'highway', 'jp', 'driving2',  # 'jp_hw','highway_normal_traffic'
                         'motorway', 'nyc', 'park',
                         'russia1', 'traffic',
                         'tw', 'tw1',  # 'tw_road', 'tw_under_bridge',
                         ])
    resol_list = ['540p', '480p', '360p', '180p']
    # resol_list = ['360p']
    # fig, axs = plt.subplots(1, figsize=(15, 10))
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.axhline(y=0.80, c='k', linestyle='--')
    # plt.axhline(y=0.85, c='k', linestyle='--')
    # plt.axhline(y=0.90, c='k', linestyle='--')
    # plt.axhline(y=0.95, c='k', linestyle='--')
    # plt.axhline(y=0.50, c='k', linestyle='--')
    # plt.ylim([-0.1, 1.1])
    # # step = 0.004
    # # area_bins = np.arange(0, 0.6, step)
    # area_bins = np.hstack([np.arange(0, 0.01, 0.001),
    #                        np.arange(0.01, 0.1, 0.01),
    #                        np.arange(0.1, 0.2, 0.05)])
    # axs.set_title('F1 vs object size using all videos at different resolutions',
    #               fontsize=15)
    # for resol2 in resol_list:
    #     print('Drawing f1 vs object size at {} ...'.format(resol2))
    #     tps, fps, fns = [], [], []
    #     for video in video_list:
    #         tp_cnt, fp_cnt, fn_cnt = plot_figs(video, resol2, area_bins)
    #         tps.append(tp_cnt)
    #         fps.append(fp_cnt)
    #         fns.append(fn_cnt)
    #     tps_tot = np.sum(tps, axis=0)
    #     fps_tot = np.sum(fps, axis=0)
    #     fns_tot = np.sum(fns, axis=0)
    #     f1_scores = []
    #     for tpos, fpos, fneg in zip(tps_tot, fps_tot, fns_tot):
    #         print(tpos + fneg)
    #         if tpos == 0 and fneg == 0:
    #             f1_scores.append(np.nan)
    #         # elif tpos + fneg < 500:
    #         #     f1_scores.append(np.nan)
    #         else:
    #             f1_scores.append(compute_f1(tpos, fpos, fneg))
    #     # for video in video_list:
    #     # plt.figure(video)
    #     axs.plot(area_bins[:-1], f1_scores, 'o-', ms=5, label='all '+resol2)
    #     np.save('{}_binwise_f1_new.npy'.format(resol2), f1_scores)
    #     np.save('{}_obj_size_bins_new.npy'.format(resol2), area_bins)
    # plt.legend()
    # axs.set_xscale('log')
    # plt.xlabel('object size', fontsize=15)
    # plt.ylabel('f1', fontsize=15)
    # plt.savefig('f1_vs_obj_sizes/f1_vs_obj_size_log.png'.format(video, resol2))

    for video in video_list:
        dt_file = os.path.join(ROOT, video, '720p', 'profile',
                               'updated_gt_FasterRCNN_COCO_no_filter.csv')
        dts, nb_frame = load_detections(video, dt_file, '720p')
        _, resol_dict, acc_dict, size_dict, cnt_dict = \
            load_awstream_profile('/data/zxxia/benchmarking/awstream/spatial_overfitting_profile_11_06/awstream_spatial_overfitting_profile_{}.csv'.format(video),
                                  '/data/zxxia/benchmarking/awstream/short_video_size.csv')
        resol2 = '360p'
        f1_scores = np.load('{}_binwise_f1_new.npy'.format(resol2))
        area_bins = np.load('{}_obj_size_bins_new.npy'.format(resol2))
        scan(dts, nb_frame, '720p', f1_scores, resol_dict,
             acc_dict, video, resol2, area_bins)
    #     plt.savefig('f1_vs_obj_sizes/{}_{}_f1.png'.format(video, resol2))

    plt.show()


def select_boxes(boxes, area_range):
    """ select boxes whose area are in area range """
    ret = list()
    for box in boxes:
        area = (box[2]-box[0])*(box[3]-box[1])/(720*1280)
        if area_range[0] <= area < area_range[1]:
            ret.append(box)
    return ret


def visualize(img, boxes, color=(0, 0, 0), mobilenet_flag=False):
    """ visulize """
    for box in boxes:
        [xmin, ymin, xmax, ymax, obj_type, score, obj_id] = box
        area = (ymax-ymin)*(xmax-xmin)/(1280*720)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        if mobilenet_flag:
            cv2.putText(img, '{:.3f}'.format(area), (xmin, ymax-10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            cv2.putText(img, '{:.3f}'.format(score), (xmin+100, ymax-10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            # cv2.putText(img, str(obj_id), (xmin, ymax+10),
            #             cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            # cv2.putText(img, str(obj_type), (xmin, ymax+10),
            #             cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        else:
            cv2.putText(img, '{:.3f}'.format(area), (xmin, ymin+20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            cv2.putText(img, '{:.3f}'.format(score), (xmin+100, ymin+20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            # cv2.putText(img, str(obj_id), (xmin, ymin-10),
            #             cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            # cv2.putText(img, str(obj_type), (xmin, ymin-10),
            #             cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)


def change_cnt(box, area_bins, cnt):
    """ change counts according to area """
    area = (box[2]-box[0])*(box[3]-box[1])/(1280*720)
    for j in range(1, len(area_bins)):
        if area_bins[j-1] <= area < area_bins[j]:
            cnt[j-1] += 1
            break


def plot_figs(video, resol2, area_bins):
    """ plot figure """
    resol1 = '720p'
    dt_file = os.path.join(ROOT, video, resol1, 'profile',
                           'updated_gt_FasterRCNN_COCO_no_filter.csv')
    dt1, nb_frame1 = load_detections(video, dt_file, resol1)

    dt_file = os.path.join(ROOT, video, resol2, 'profile',
                           'updated_gt_FasterRCNN_COCO_no_filter.csv')
    dt2, nb_frame2 = load_detections(video, dt_file, resol2)
    nb_frame1 = min(nb_frame1, nb_frame2)
    nb_frame2 = min(nb_frame1, nb_frame2)
    assert nb_frame1 == nb_frame2, '{} != {}'.format(nb_frame1, nb_frame2)
    iou_thresh = 0.5
    tp_cnt = np.zeros(len(area_bins)-1)
    fp_cnt = np.zeros(len(area_bins)-1)
    fn_cnt = np.zeros(len(area_bins)-1)

    for i in range(1, nb_frame1+1):
        scaled_boxes2 = scale_boxes(dt2[i], RESOL_DICT[resol2],
                                    RESOL_DICT[resol1])
        gt_idx_thr = []
        pred_idx_thr = []
        ious = []
        for igb, gt_box in enumerate(dt1[i]):
            for ipb, pred_box in enumerate(scaled_boxes2):
                iou = IoU(pred_box, gt_box)
                if iou > iou_thresh:
                    gt_idx_thr.append(igb)
                    pred_idx_thr.append(ipb)
                    ious.append(iou)
        args_desc = np.argsort(ious)[::-1]
        gt_match_idx = []
        pred_match_idx = []
        if len(args_desc) == 0:
            # No matches
            # tpos = 0
            # fpos = len(pred_boxes)
            # fneg = len(gt_boxes)

            for box in scaled_boxes2:
                change_cnt(box, area_bins, fp_cnt)
            for box in dt1[i]:
                change_cnt(box, area_bins, fn_cnt)
        else:
            for idx in args_desc:
                gt_idx = gt_idx_thr[idx]
                pr_idx = pred_idx_thr[idx]
                # If the boxes are unmatched, add them to matches
                if (gt_idx not in gt_match_idx) and \
                        (pr_idx not in pred_match_idx):
                    gt_match_idx.append(gt_idx)
                    pred_match_idx.append(pr_idx)
                    gt_box = dt1[i][gt_idx]
                    area = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1]) / \
                        (720*1280)
                    for j in range(1, len(area_bins)):
                        if area_bins[j-1] <= area < area_bins[j]:
                            tp_cnt[j-1] += 1
                            break
            for box_idx, box in enumerate(scaled_boxes2):
                if box_idx in pred_match_idx:
                    continue
                change_cnt(box, area_bins, fp_cnt)
            for box_idx, box in enumerate(dt1[i]):
                if box_idx in gt_match_idx:
                    continue
                change_cnt(box, area_bins, fn_cnt)
        # save_flag = True
        save_flag = False
        tp_boxes2plot1 = []
        fn_boxes2plot1 = []
        tp_boxes2plot2 = []
        fp_boxes2plot2 = []
        for box_idx, box in enumerate(dt1[i]):
            if box_idx not in gt_match_idx:
                [xmin, ymin, xmax, ymax, obj_type, score, obj_id] = box
                area = (ymax-ymin)*(xmax-xmin)/(1280*720)
                # if 0.178 <= area < 0.188:
                # if 0.04 <= area < 0.08:
                #     save_flag = True
                fn_boxes2plot1.append(box)
            else:
                tp_boxes2plot1.append(box)
        for box_idx, box in enumerate(scaled_boxes2):
            if box_idx not in pred_match_idx:
                [xmin, ymin, xmax, ymax, obj_type, score, obj_id] = box
                area = (ymax-ymin)*(xmax-xmin)/(1280*720)
                # if 0.178 <= area < 0.188:
                # if 0.04 <= area < 0.08:
                #     save_flag = True
                fp_boxes2plot2.append(box)
            else:
                tp_boxes2plot2.append(box)
        if save_flag:
            img_name = os.path.join(DATA_PATH, video, resol1,
                                    '{:06d}.jpg'.format(i))
            img = cv2. imread(img_name)
            visualize(img, tp_boxes2plot1, color=(0, 255, 0),
                      mobilenet_flag=False)
            visualize(img, fn_boxes2plot1, color=(0, 255, 255),
                      mobilenet_flag=False)
            visualize(img, tp_boxes2plot2, color=(255, 0, 0),
                      mobilenet_flag=True)
            visualize(img, fp_boxes2plot2, color=(0, 0, 255),
                      mobilenet_flag=True)
            # cv2.imshow(img_name, img)
            # cv2.moveWindow(img_name, 200, 200)
            print('save image {}...'.format(i))
            cv2.imwrite(os.path.join('russia1_vis',
                                     '{:06d}.jpg'.format(i)), img)
        # if i > 10:
            # cv2.destroyWindow(format(i-10, '06d') + '.jpg')
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     # args.visualize = False
            #     cv2.destroyAllWindows()
            #     break

    f1_scores = []
    for tpos, fpos, fneg in zip(tp_cnt, fp_cnt, fn_cnt):
        if tpos == 0 and fneg == 0:
            f1_scores.append(np.nan)
        elif tpos + fneg < 500:
            f1_scores.append(np.nan)
        else:
            f1_scores.append(compute_f1(tpos, fpos, fneg))
    # plt.figure(video, figsize=(15, 10))
    # plt.plot(area_bins[:-1], f1_scores, 'o-', ms=5, label=video+' '+resol2)
    # plt.title(video+resol2, fontsize=15)
    # if video == 'russia1':
    #     for tpos, fpos, fneg, area, f1_score in zip(tp_cnt, fp_cnt, fn_cnt,
    #                                                 area_bins[:-1]+step/2,
    #                                                 f1_scores):
    #         plt.annotate("({},{},{})"
    #                      .format(int(tpos), int(fpos), int(fneg)),
    #                      (area, f1_score), fontsize=15)
    # plt.ylim([-0.1, 1.1])
    # plt.xlim([0, 0.2])
    # plt.xlabel('object size', fontsize=15)
    # plt.ylabel('f1', fontsize=15)
    # print(tp_cnt)
    # print(f1_scores)
    return tp_cnt, fp_cnt, fn_cnt


def scan(dts1, nb_frame, resol, benchmark_f1,
         resol_dict, acc_dict, video, resol2, area_bins):

    short_video_length = 30
    fps = 30
    nb_short_videos = nb_frame//(short_video_length*fps)
    gt_f1 = []
    scan_f1 = []
    scan_good_cnt = 0
    scan_bad_cnt = 0
    good_cnt = 0
    bad_cnt = 0
    interest_sid = list()
    for i in range(nb_short_videos):
        clip = video + '_' + str(i)
        start = i*short_video_length*fps+1
        end = (i+1)*short_video_length*fps
        areas = []
        cnt = np.zeros(len(area_bins)-1)
        for frame_idx in range(start, end+1):
            areas.extend([compute_area(box)/(RESOL_DICT[resol][0]*RESOL_DICT[resol][1])
                         for box in dts1[frame_idx]])
        if len(areas) < 500:
            continue
        for area in areas:
            for j in range(1, len(area_bins)):
                if area_bins[j-1] <= area < area_bins[j]:
                    cnt[j-1] += 1
                    break
        cnt_percent = cnt/np.sum(cnt)
        pred_f1 = np.dot(np.nan_to_num(benchmark_f1),
                         np.nan_to_num(cnt_percent))
        scan_f1.append(pred_f1)

        if pred_f1 >= 0.85:
            scan_good_cnt += 1
        if pred_f1 <= 0.50:
            scan_bad_cnt += 1
        for res, acc in zip(resol_dict[clip], acc_dict[clip]):
            if res == RESOL_DICT[resol2][1]:
                gt_f1.append(acc)
                if acc >= 0.85:
                    good_cnt += 1
                if acc <= 0.50:
                    bad_cnt += 1
        interest_sid.append(i)
    if not interest_sid:
        return

    print('{}:\n\toriginal good percent={}, bad percent={}'
          .format(video, good_cnt/len(scan_f1), bad_cnt/len(scan_f1)))
    print('\tscan good percent={}, bad percent={}'
          .format(scan_good_cnt/len(scan_f1), scan_bad_cnt/len(scan_f1)))

    plt.figure(figsize=(15, 10))
    plt.plot(interest_sid, scan_f1, 'o-', label='scanned f1')
    plt.plot(interest_sid, gt_f1, 'o-', label='groundtruth f1')
    plt.title(video+' at '+resol2, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim([0, 1])
    plt.xlabel('30s-clip index', fontsize=15)
    plt.ylabel('f1', fontsize=15)
    plt.legend()



if __name__ == '__main__':
    main()
