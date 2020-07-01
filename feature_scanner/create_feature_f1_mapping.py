import copy
import os
import numpy as np
from benchmarking.awstream.Awstream import scale_boxes
from benchmarking.feature_analysis.features import compute_velocity, \
    compute_box_size  # , compute_video_object_size
from benchmarking.video import YoutubeVideo  # , WaymoVideo
from benchmarking.utils.utils import IoU, compute_f1

DT_ROOT = '/data/zxxia/benchmarking/results/videos'
# VIDEOS = ['crossroad', 'crossroad2', 'crossroad3', 'crossroad4', 'drift',
#           'driving1', 'driving_downtown', 'highway',
#           'nyc', 'jp',  'lane_split',  'driving2',
#           'motorway', 'park', 'russia', 'russia1', 'traffic', 'tw', 'tw1',
#           'tw_under_bridge']
VIDEOS = sorted(['crossroad', 'crossroad2', 'crossroad3', 'crossroad4',
                 'driving1', 'driving_downtown',  # 'lane_split',
                 'highway', 'driving2',  # 'jp', drift
                 # 'jp_hw','highway_normal_traffic'
                 'motorway', 'nyc', 'park',
                 'russia1',
                 'traffic',
                 # 'tw', 'tw1',  # 'tw_road', 'tw_under_bridge',
                 # 'road_trip'
                 ])

RESOL_LIST = ['480p']  # , '180p']
TEMPORAL_SAMPLING_LIST = [2.5]


def change_cnt(val, bins, cnt):
    """Change count according to val."""
    for j in range(1, len(bins)):
        if bins[j-1] <= val < bins[j]:
            cnt[j-1] += 1
            break


def compute_binwise_f1(tp_list, fp_list, fn_list):
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


def object_velocity_f1_mapping(video, velocity, sample_rate, velocity_bins,
                               iou_thresh=0.5):
    nb_frame = min(video.frame_count, video.frame_count)
    tp_cnt = np.zeros(len(velocity_bins)-1)
    fp_cnt = np.zeros(len(velocity_bins)-1)
    fn_cnt = np.zeros(len(velocity_bins)-1)

    temporal_sampled_dets = {}
    save_dt = []

    for img_index in range(1,  nb_frame + 1):
        dt_boxes_final = []
        current_full_model_dt = video.get_frame_detection(img_index)
        # based on sample rate, decide whether this frame is sampled
        if img_index % sample_rate >= 1:
            # this frame is not sampled, so reuse the last saved
            # detection result
            dt_boxes_final = copy.deepcopy(save_dt)
        else:
            # this frame is sampled, so use the full model result
            dt_boxes_final = copy.deepcopy(current_full_model_dt)
            save_dt = copy.deepcopy(dt_boxes_final)
        temporal_sampled_dets[img_index] = dt_boxes_final

        # tpos[img_index], fpos[img_index], fneg[img_index] = \
        #     eval_single_image(current_gt, dt_boxes_final)

    for i in range(1, nb_frame+1):
        if not velocity[i]:
            continue
        temporal_sampled_boxes = temporal_sampled_dets[i]
        gt_idx_thr = []
        pred_idx_thr = []
        ious = []
        for igb, gt_box in enumerate(video.get_frame_detection(i)):
            for ipb, pred_box in enumerate(temporal_sampled_boxes):
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

            for box in temporal_sampled_boxes:
                # TODO: fix this
                pass
                # change_cnt(velo, velocity_bins, fp_cnt)
            for box, velo in zip(video.get_frame_detection(i), velocity[i]):
                change_cnt(velo, velocity_bins, fn_cnt)
        else:
            for idx in args_desc:
                gt_idx = gt_idx_thr[idx]
                pr_idx = pred_idx_thr[idx]
                # If the boxes are unmatched, add them to matches
                if (gt_idx not in gt_match_idx) and \
                        (pr_idx not in pred_match_idx):
                    gt_match_idx.append(gt_idx)
                    pred_match_idx.append(pr_idx)
                    # gt_box = video.get_frame_detection(i)[gt_idx]

                    try:
                        change_cnt(velocity[i][gt_idx], velocity_bins, tp_cnt)
                    except IndexError:
                        import pdb
                        pdb.set_trace()
            for box_idx, box in enumerate(temporal_sampled_boxes):
                if box_idx in pred_match_idx:
                    continue
                # TODO: fix this
                pass
                # change_cnt(velo, velocity_bins, fp_cnt)
            for box_idx, box, velo in enumerate(
                    zip(video.get_frame_detection(i), velocity)):
                if box_idx in gt_match_idx:
                    continue
                change_cnt(velo, velocity_bins, fn_cnt)

    return tp_cnt, fp_cnt, fn_cnt


def object_size_f1_mapping(original_video, video, area_bins, iou_thresh=0.5):
    # assert original_video.frame_count == video.frame_count, '{} != {}'.
    # format( #     original_video.frame_count, video.frame_count)
    nb_frame = min(original_video.frame_count, video.frame_count)
    tp_cnt = np.zeros(len(area_bins)-1)
    fp_cnt = np.zeros(len(area_bins)-1)
    fn_cnt = np.zeros(len(area_bins)-1)

    for i in range(1, nb_frame+1):
        scaled_boxes = scale_boxes(video.get_frame_detection(i),
                                   video.resolution,
                                   original_video.resolution)
        gt_idx_thr = []
        pred_idx_thr = []
        ious = []
        for igb, gt_box in enumerate(original_video.get_frame_detection(i)):
            for ipb, pred_box in enumerate(scaled_boxes):
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

            for box in scaled_boxes:
                area = compute_box_size(box) / (original_video.resolution[0]
                                                * original_video.resolution[1])
                change_cnt(area, area_bins, fp_cnt)
                # change_cnt(box, original_video.resolution, area_bins, fp_cnt)
            for box in original_video.get_frame_detection(i):
                area = compute_box_size(box) / (original_video.resolution[0]
                                                * original_video.resolution[1])
                change_cnt(area, area_bins, fn_cnt)
                # change_cnt(box, original_video.resolution, area_bins, fn_cnt)
        else:
            for idx in args_desc:
                gt_idx = gt_idx_thr[idx]
                pr_idx = pred_idx_thr[idx]
                # If the boxes are unmatched, add them to matches
                if (gt_idx not in gt_match_idx) and \
                        (pr_idx not in pred_match_idx):
                    gt_match_idx.append(gt_idx)
                    pred_match_idx.append(pr_idx)
                    gt_box = original_video.get_frame_detection(i)[gt_idx]
                    area = compute_box_size(gt_box) / \
                        (original_video.resolution[0]
                         * original_video.resolution[1])
                    change_cnt(area, area_bins, tp_cnt)
                    # for j in range(1, len(area_bins)):
                    #     if area_bins[j-1] <= area < area_bins[j]:
                    #         tp_cnt[j-1] += 1
                    #         break
            for box_idx, box in enumerate(scaled_boxes):
                if box_idx in pred_match_idx:
                    continue
                area = compute_box_size(box) / (original_video.resolution[0]
                                                * original_video.resolution[1])
                change_cnt(area, area_bins, fp_cnt)
            for box_idx, box in enumerate(
                    original_video.get_frame_detection(i)):
                if box_idx in gt_match_idx:
                    continue
                area = compute_box_size(box) / (original_video.resolution[0]
                                                * original_video.resolution[1])
                change_cnt(area, area_bins, fn_cnt)

    return tp_cnt, fp_cnt, fn_cnt


def main():
    # step = 0.004
    # area_bins = np.arange(0, 1, step)
    # for resol in RESOL_LIST:
    #     tot_tp_cnt = np.zeros(len(area_bins)-1)
    #     tot_fp_cnt = np.zeros(len(area_bins)-1)
    #     tot_fn_cnt = np.zeros(len(area_bins)-1)
    #     for name in VIDEOS:
    #         print('load', resol, name)
    #         dt_file = os.path.join(DT_ROOT, name, '720p', 'profile',
    #                                'updated_gt_FasterRCNN_COCO_no_filter.csv')
    #         metadata_file = '/data/zxxia/videos/{}/metadata.json'.format(name)
    #         original_video = YoutubeVideo(
    #             name, '720p', metadata_file, dt_file, None)
    #         dt_file = os.path.join(DT_ROOT, name, resol, 'profile',
    #                                'updated_gt_FasterRCNN_COCO_no_filter.csv')
    #         metadata_file = '/data/zxxia/videos/{}/metadata.json'.format(name)
    #         video = YoutubeVideo(name, resol, metadata_file, dt_file, None)
    #         tp_cnt, fp_cnt, fn_cnt = object_size_f1_mapping(
    #             original_video, video, area_bins)
    #
    #         # print(tp_cnt, fp_cnt, fn_cnt)
    #         tot_tp_cnt += tp_cnt
    #         tot_fp_cnt += fp_cnt
    #         tot_fn_cnt += fn_cnt
    #     # save the benchmark object size and f1
    #     binwise_f1 = compute_binwise_f1(tot_tp_cnt, tot_fp_cnt, tot_fn_cnt)
    #     np.save('f1_vs_obj_sizes/{}_binwise_f1.npy'.format(resol), binwise_f1)
    #     np.save('f1_vs_obj_sizes/{}_obj_size_bins.npy'.format(resol), area_bins)

    step = 0.01
    velocity_bins = np.arange(0, 3, step)
    for sample_rate in TEMPORAL_SAMPLING_LIST:
        tot_tp_cnt = np.zeros(len(velocity_bins)-1)
        tot_fp_cnt = np.zeros(len(velocity_bins)-1)
        tot_fn_cnt = np.zeros(len(velocity_bins)-1)
        for name in VIDEOS:
            print('load', sample_rate, name)
            dt_file = os.path.join(DT_ROOT, name, '720p', 'profile',
                                   'updated_gt_FasterRCNN_COCO_no_filter.csv')
            metadata_file = '/data/zxxia/videos/{}/metadata.json'.format(name)
            original_video = YoutubeVideo(
                name, '720p', metadata_file, dt_file, None)
            velocity = compute_velocity(
                original_video.get_video_detection(),
                original_video.start_frame_index,
                original_video.end_frame_index, original_video.frame_rate)
            tp_cnt, fp_cnt, fn_cnt = object_velocity_f1_mapping(
                original_video, velocity, sample_rate, velocity_bins)

            # print(tp_cnt, fp_cnt, fn_cnt)
            tot_tp_cnt += tp_cnt
            tot_fp_cnt += fp_cnt
            tot_fn_cnt += fn_cnt
        # save the benchmark object size and f1
        binwise_f1 = compute_binwise_f1(tot_tp_cnt, tot_fp_cnt, tot_fn_cnt)
        np.save('f1_vs_obj_sizes/{}_binwise_f1.npy'.format(sample_rate),
                binwise_f1)
        np.save('f1_vs_obj_sizes/{}_velocity_bins.npy'.format(sample_rate),
                velocity_bins)


if __name__ == '__main__':
    main()
