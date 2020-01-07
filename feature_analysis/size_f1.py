""" object size vs accuracy """
import os
# import pdb
import numpy as np
import matplotlib.pyplot as plt
import cv2
from benchmarking.constants import CAMERA_TYPES, COCOLabels, RESOL_DICT
from benchmarking.utils.utils import IoU, compute_f1, load_metadata
from benchmarking.utils.model_utils import load_full_model_detection, \
    filter_video_detections, remove_overlappings
from benchmarking.awstream.Awstream import scale_boxes
from benchmarking.feature_analysis.helpers import load_awstream_profile, sample_frames, \
    get_areas, plot_cdf

ROOT = '/data/zxxia/benchmarking/results/videos'
VIDEO_TO_DELETE = ['crossroad', 'nyc', 'russia', 'crossroad2',
                   'driving_downtown',
                   'tw_road', 'tw_under_bridge', 'tw1', 'tw', 'crossroad3']

DATA_PATH = '/data/zxxia/videos'
DATA_PATH2 = '/data2/zxxia/videos'
VIDEO_LIST = sorted(['crossroad', 'crossroad2', 'crossroad3', 'crossroad4',
                     'driving1', 'driving_downtown',  # 'lane_split',
                     'highway', 'driving2',  # 'jp', drift
                     # 'jp_hw','highway_normal_traffic'
                     'motorway', 'nyc', 'park',
                     'russia1',
                     'traffic',
                     # 'tw', 'tw1',  # 'tw_road', 'tw_under_bridge',
                     # 'road_trip'
                     ])
# VIDEO_LIST = ['driving_downtown']
SHORT_VIDEO_LENGTH = 30

GOOD_CASE_TH = 0.90
BAD_CASE_TH = 0.60


def main():
    """ main """
    resol_list = ['540p', '480p', '360p']  # , '180p']
    resol_list = ['480p']
    fig, axs = plt.subplots(1, figsize=(15, 10))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.axhline(y=0.80, c='k', linestyle='--')
    plt.axhline(y=0.85, c='k', linestyle='--')
    # plt.axhline(y=0.90, c='k', linestyle='--')
    # plt.axhline(y=0.95, c='k', linestyle='--')
    plt.axhline(y=0.50, c='k', linestyle='--')
    # plt.ylim([-0.1, 1.1])
    step = 0.004
    area_bins = np.arange(0, 1, step)
    # area_bins = np.hstack([np.arange(0, 0.01, 0.001),
    #                        np.arange(0.01, 0.1, 0.01),
    #                        np.arange(0.1, 0.6, 0.05)])
    axs.set_title('F1 vs object size using all videos at different resols',
                  fontsize=15)
    # for resol2 in resol_list:
    #     print('Drawing f1 vs object size at {} ...'.format(resol2))
    #     tps, fps, fns = [], [], []
    #     for video in VIDEO_LIST:
    #         tp_cnt, fp_cnt, fn_cnt = profile(video, resol2, area_bins)
    #         f1_scores = binwise_f1(tp_cnt, fp_cnt, fn_cnt)
    #         # axs.plot(area_bins[:-1], f1_scores, 'o-', ms=5, label=video+resol2)
    #         # print(video, tp_cnt + fn_cnt)
    #         # axs.plot(area_bins[:-1], f1_scores, 'o-', ms=5, label=video+resol2)
    #         tps.append(tp_cnt)
    #         fps.append(fp_cnt)
    #         fns.append(fn_cnt)
    #         if video == 'driving_downtown':
    #             axs.plot(area_bins[:-1], f1_scores,
    #                      '^-', ms=5, label=video+resol2)
    #     tps_tot = np.sum(tps, axis=0)
    #     fps_tot = np.sum(fps, axis=0)
    #     fns_tot = np.sum(fns, axis=0)
    #     # print('all', tps_tot + fns_tot)
    #     f1_scores_all = binwise_f1(tps_tot, fps_tot, fns_tot)
    #
    #     axs.plot(area_bins[:-1], f1_scores_all,
    #              'x-', ms=5, label='all '+resol2)
    #     np.save('{}_binwise_f1.npy'.format(resol2), f1_scores)
    #     np.save('{}_obj_size_bins.npy'.format(resol2), area_bins)

    # plt.legend()
    # # axs.set_xscale('log')
    # plt.xlabel('object size', fontsize=15)
    # plt.ylabel('f1', fontsize=15)
    # plt.savefig('f1_vs_obj_sizes/f1_vs_obj_size_log.png'.format(video, resol2))

    # video_list = ['driving1']
    video_list = ['driving_downtown']
    # video_list = ['park']
    for video in video_list:
        metadata_file = '/data/zxxia/videos/{}/metadata.json'.format(video)
        metadata = load_metadata(metadata_file)
        dt_file = os.path.join(ROOT, video, '720p', 'profile',
                               'updated_gt_FasterRCNN_COCO_no_filter.csv')
        dts, _ = load_detections(video, dt_file, '720p')

        dt_file = os.path.join(ROOT, video, '720p', 'profile',
                               'updated_gt_mobilenet_COCO_no_filter.csv')
        dts_mobilenet, _ = load_detections(video, dt_file, '720p')

        if SHORT_VIDEO_LENGTH == 0:
            _, resol_dict, acc_dict, _, _ = \
                load_awstream_profile('/data/zxxia/benchmarking/awstream/'
                                      'spatial_overfitting_profile_11_06/'
                                      'awstream_spatial_overfitting_profile_{}.csv'
                                      .format(video),
                                      '/data/zxxia/benchmarking/awstream/'
                                      'short_video_size.csv')
        else:  # SHORT_VIDEO_LENGTH == 10:
            _, resol_dict, acc_dict, _, _ = \
                load_awstream_profile('/data/zxxia/benchmarking/awstream/'
                                      'spatial_overfitting_profile_{}s/'
                                      'awstream_spatial_overfitting_profile_{}.csv'
                                      .format(SHORT_VIDEO_LENGTH, video),
                                      '/data/zxxia/benchmarking/awstream/'
                                      'short_video_size.csv')
        resol2 = '480p'
        f1_scores_all = np.load('{}_binwise_f1.npy'.format(resol2))
        area_bins = np.load('{}_obj_size_bins.npy'.format(resol2))

        interest_sid, gt_f1, scan_f1, scan_f1_mobilenet, \
            scan_f1_mobilenet_sampled = scan(dts, dts_mobilenet, metadata,
                                             '720p', f1_scores_all,
                                             resol_dict, acc_dict, video,
                                             resol2, area_bins)

        plt.figure(figsize=(15, 10))
        assert len(interest_sid) == len(scan_f1)
        assert len(interest_sid) == len(scan_f1_mobilenet_sampled)
        assert len(interest_sid) == len(scan_f1_mobilenet)
        assert len(interest_sid) == len(gt_f1), str(
            len(interest_sid)) + ' ' + str(len(gt_f1))
        plt.plot(interest_sid, scan_f1, 'o-', label='frcnn scanned f1')
        plt.plot(interest_sid, gt_f1, 'o-', label='groundtruth f1')
        plt.plot(interest_sid, scan_f1_mobilenet, 'o-',
                 label='mobilenet scanned f1')
        plt.plot(interest_sid, scan_f1_mobilenet_sampled, 'o-',
                 label='sampled mobilenet groundtruth f1')
        plt.title(video+' at '+resol2, fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([0, 1])
        plt.xlabel('{}s-clip index'.format(SHORT_VIDEO_LENGTH), fontsize=15)
        plt.ylabel('f1', fontsize=15)
        plt.legend()
    #     plt.savefig('f1_vs_obj_sizes/{}_{}_f1.png'.format(video, resol2))
    plt.show()


def binwise_f1(tp_list, fp_list, fn_list):
    """Compute f1 score within bins."""
    assert len(tp_list) == len(fp_list)
    assert len(fp_list) == len(fn_list)
    f1_scores = []

    for tpos, fpos, fneg in zip(tp_list, fp_list, fn_list):
        if tpos == 0 and fneg == 0:
            f1_scores.append(np.nan)
        # elif tpos + fneg < 500:
        #     f1_scores.append(np.nan)
        else:
            f1_scores.append(compute_f1(tpos, fpos, fneg))
    return f1_scores


def load_detections(video, dt_file, resol):
    """ load and filter detections """
    dts, nb_frame = load_full_model_detection(dt_file)
    if video in CAMERA_TYPES['moving']:
        dts, _ = filter_video_detections(dts,
                                         target_types={COCOLabels.CAR.value,
                                                       COCOLabels.BUS.value,
                                                       # COCOLabels.TRAIN.value,
                                                       COCOLabels.TRUCK.value
                                                       },
                                         height_range=(RESOL_DICT[resol][1]//20,
                                                       RESOL_DICT[resol][1]),
                                         )  # score_range=(0.5, 1)
    else:
        dts, _ = filter_video_detections(dts,
                                         target_types={COCOLabels.CAR.value,
                                                       COCOLabels.BUS.value,
                                                       # COCOLabels.TRAIN.value,
                                                       COCOLabels.TRUCK.value
                                                       },
                                         width_range=(
                                             0, RESOL_DICT[resol][0]/2),
                                         height_range=(RESOL_DICT[resol][0]//20,
                                                       RESOL_DICT[resol][0]/2))
    if video == 'road_trip':
        for frame_idx in dts:
            tmp_boxes = []
            for box in dts[frame_idx]:
                xmin, ymin, xmax, ymax = box[:4]
                if ymin >= 500/720*RESOL_DICT[resol][1] or ymax >= 645/720*RESOL_DICT[resol][1]:
                    continue
                if (xmax - xmin) >= 2/3 * RESOL_DICT[resol][0]:
                    continue
                tmp_boxes.append(box)
            dts[frame_idx] = tmp_boxes
    for frame_idx, bboxes in dts.items():
        # merge all vehicle labels into CAR
        for box_pos, box in enumerate(bboxes):
            box[4] = COCOLabels.CAR.value
            bboxes[box_pos] = box
        dts[frame_idx] = bboxes
        # remove overlappings to mitigate occultation
        dts[frame_idx] = remove_overlappings(bboxes, 0.3)

    return dts, nb_frame


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
            cv2.putText(img, str(obj_type), (xmin+200, ymax-10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
        else:
            cv2.putText(img, '{:.3f}'.format(area), (xmin, ymin+20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            cv2.putText(img, '{:.3f}'.format(score), (xmin+100, ymin+20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            # cv2.putText(img, str(obj_id), (xmin, ymin-10),
            #             cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            cv2.putText(img, str(obj_type), (xmin+200, ymin+20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)


def change_cnt(box, area_bins, cnt):
    """ change counts according to area """
    area = (box[2]-box[0])*(box[3]-box[1])/(1280*720)
    for j in range(1, len(area_bins)):
        if area_bins[j-1] <= area < area_bins[j]:
            cnt[j-1] += 1
            break


def profile(video, resol2, area_bins, iou_thresh=0.5):
    """plot figure."""
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
                # if 0.09 <= area < 0.1:
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
                # if 0.09 <= area < 0.1:
                #     save_flag = True
                fp_boxes2plot2.append(box)
            else:
                tp_boxes2plot2.append(box)
        if save_flag:
            img_name = os.path.join(DATA_PATH2, video, resol1,
                                    '{:06d}.jpg'.format(i))
            img = cv2. imread(img_name)
            # visualize(img, tp_boxes2plot1, color=(0, 255, 0),
            #           mobilenet_flag=False)
            visualize(img, fn_boxes2plot1, color=(0, 255, 255),
                      mobilenet_flag=False)
            # visualize(img, tp_boxes2plot2, color=(255, 0, 0),
            #           mobilenet_flag=True)
            visualize(img, fp_boxes2plot2, color=(0, 0, 255),
                      mobilenet_flag=True)
            # cv2.imshow(img_name, img)
            # cv2.moveWindow(img_name, 200, 200)
            print('save image {}...'.format(i))
            cv2.imwrite(os.path.join('road_trip_vis',
                                     '{:06d}.jpg'.format(i)), img)
        # if i > 10:
            # cv2.destroyWindow(format(i-10, '06d') + '.jpg')
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     # args.visualize = False
            #     cv2.destroyAllWindows()
            #     break

    return tp_cnt, fp_cnt, fn_cnt


def scan(dts, dts_mobilenet, metadata, resol, benchmark_f1,
         resol_dict, acc_dict, video, resol2, area_bins):
    """ do feature scanning on object size """
    nframe = 10
    dts_mobilenet_sampled = sample_frames(dts_mobilenet,
                                          metadata['frame count'], nframe)
    # print(len(dts_mobilenet), len(dts_mobilenet_sampled))
    # areas = get_areas(dts, 1, metadata['frame count'], resol)
    # areas_sampled = get_areas(dts, 1, metadata['frame count'], resol)
    # areas_mobilenet = get_areas(
    #     dts_mobilenet, 1, metadata['frame count'], resol)
    # areas_mobilenet_sampled = get_areas(dts_mobilenet_sampled, 1,
    #                                     metadata['frame count'], resol)
    #
    # plot_cdf(areas, 1000, 'frcnn')
    # plot_cdf(areas_sampled, 1000, 'frcnn sampled {}'.format(nframe))
    # plot_cdf(areas_mobilenet, 1000, 'mobilenet')
    # plot_cdf(areas_mobilenet_sampled, 1000,
    #          'mobilenet sampled {}'.format(nframe))
    # plt.legend()
    # plt.show()
    nb_short_videos = metadata['frame count']//(
        SHORT_VIDEO_LENGTH*metadata['frame rate'])
    print(nb_short_videos)
    gt_f1 = []
    scan_f1 = []
    scan_f1_mobilenet = []
    scan_f1_mobilenet_sampled = []

    scan_good_cnt = 0
    scan_bad_cnt = 0

    scan_good_cnt_mobilenet = 0
    scan_bad_cnt_mobilenet = 0

    scan_good_cnt_mobilenet_sampled = 0
    scan_bad_cnt_mobilenet_sampled = 0
    good_cnt = 0
    bad_cnt = 0
    baseline_good_cnt = 0
    baseline_bad_cnt = 0
    interest_sid = list()
    for i in range(nb_short_videos):
        clip = video + '_' + str(i)
        start = i*SHORT_VIDEO_LENGTH*metadata['frame rate']+1
        end = (i+1)*SHORT_VIDEO_LENGTH*metadata['frame rate']
        cnt = np.zeros(len(area_bins)-1)
        cnt_mobilenet = np.zeros(len(area_bins)-1)
        cnt_mobilenet_sampled = np.zeros(len(area_bins)-1)
        areas = get_areas(dts, start, end, resol)
        areas_mobilenet = get_areas(dts_mobilenet, start, end, resol)
        areas_mobilenet_sampled = get_areas(dts_mobilenet_sampled,
                                            start, end, resol)
        # print(clip, 'nb of box:', len(areas))
        # if len(areas) < 500:
        #     continue
        for area in areas:
            for j in range(1, len(area_bins)):
                if area_bins[j-1] <= area < area_bins[j]:
                    cnt[j-1] += 1
                    break

        for area in areas_mobilenet:
            for j in range(1, len(area_bins)):
                if area_bins[j-1] <= area < area_bins[j]:
                    cnt_mobilenet[j-1] += 1
                    break

        for area in areas_mobilenet_sampled:
            for j in range(1, len(area_bins)):
                if area_bins[j-1] <= area < area_bins[j]:
                    cnt_mobilenet_sampled[j-1] += 1
                    break

        cnt_percent = cnt/np.sum(cnt)
        pred_f1 = np.dot(np.nan_to_num(benchmark_f1),
                         np.nan_to_num(cnt_percent))
        scan_f1.append(pred_f1)

        cnt_percent_mobilenet = cnt_mobilenet/np.sum(cnt_mobilenet)
        pred_f1_mobilenet = np.dot(np.nan_to_num(benchmark_f1),
                                   np.nan_to_num(cnt_percent_mobilenet))
        scan_f1_mobilenet.append(pred_f1_mobilenet)

        # print(np.sum(cnt_mobilenet_sampled))
        cnt_percent_mobilenet_sampled = cnt_mobilenet_sampled / \
            np.sum(cnt_mobilenet_sampled)
        pred_f1_mobilenet_sampled = np.dot(np.nan_to_num(benchmark_f1),
                                           np.nan_to_num(cnt_percent_mobilenet_sampled))
        scan_f1_mobilenet_sampled.append(pred_f1_mobilenet_sampled)

        # use groundtruth
        for res, acc in zip(resol_dict[clip], acc_dict[clip]):
            if res == RESOL_DICT[resol2][1]:
                gt_f1.append(acc)
                if acc >= GOOD_CASE_TH:
                    good_cnt += 1
                elif acc <= BAD_CASE_TH:
                    bad_cnt += 1

        # use FasterRCNN computed features to scan
        if pred_f1 >= GOOD_CASE_TH:
            scan_good_cnt += 1
        elif pred_f1 <= BAD_CASE_TH:
            scan_bad_cnt += 1

        # use Mobilenet computed features to scan
        if pred_f1_mobilenet >= GOOD_CASE_TH:
            scan_good_cnt_mobilenet += 1
        elif pred_f1_mobilenet <= BAD_CASE_TH:
            scan_bad_cnt_mobilenet += 1

        # use Mobilenet computed features to scan
        if pred_f1_mobilenet_sampled >= GOOD_CASE_TH:
            scan_good_cnt_mobilenet_sampled += 1
        elif pred_f1_mobilenet_sampled <= BAD_CASE_TH:
            scan_bad_cnt_mobilenet_sampled += 1

        # use Mobilenet scanned computed features to scan
        interest_sid.append(i)
    # baseline
    nb_baseline_videos = len(gt_f1)//nframe
    print('nb of baseline', nb_baseline_videos)
    print(gt_f1[158:158 + nb_baseline_videos])
    print(gt_f1[77:77 + nb_baseline_videos])
    for acc in gt_f1[0:0 + nb_baseline_videos]:
        if acc >= GOOD_CASE_TH:
            baseline_good_cnt += 1
        elif acc <= BAD_CASE_TH:
            baseline_bad_cnt += 1
    if not interest_sid:
        return

    print('{}:\n\toriginal good percent={}, bad percent={}'
          .format(video, good_cnt/len(interest_sid),
                  bad_cnt/len(interest_sid)))
    print('\tFaserRCNN scan good percent={}, bad percent={}'
          .format(scan_good_cnt/len(interest_sid),
                  scan_bad_cnt/len(interest_sid)))
    print('\tMobilenet scan good percent={}, bad percent={}'
          .format(scan_good_cnt_mobilenet/len(interest_sid),
                  scan_bad_cnt_mobilenet/len(interest_sid)))
    print('\tSampled(per {}) Mobilenet scan good percent={}, bad percent={}'
          .format(nframe, scan_good_cnt_mobilenet_sampled/len(interest_sid),
                  scan_bad_cnt_mobilenet_sampled/len(interest_sid)))
    if nb_baseline_videos > 0:
        print('\tbaseline good percent={}, bad percent={}'
              .format(baseline_good_cnt/nb_baseline_videos,
                      baseline_bad_cnt/nb_baseline_videos))

    return interest_sid, gt_f1, scan_f1, scan_f1_mobilenet, scan_f1_mobilenet_sampled


if __name__ == '__main__':
    main()
