import cv2
# import time
import numpy as np
from collections import defaultdict
# import os
from utils.utils import interpolation, compute_f1
from utils.model_utils import eval_single_image, filter_video_detections, filter_frame_detections
from matplotlib import pyplot as plt
import pdb

DEBUG = False
# DEBUG = True


def debug_print(msg):
    if DEBUG:
        print(msg)


def frame_difference(old_frame, new_frame, bboxes_last_triggered, bboxes, thresh=35):
    '''
    Compute the number of pixel value differences which are greater than thresh
    '''
    # thresh = 35 is used in Glimpse paper
    diff = np.absolute(new_frame.astype(int) - old_frame.astype(int))
    mask = np.greater(diff, thresh)
    pix_change_obj = 0
    obj_region = np.zeros_like(new_frame)
    for box in bboxes_last_triggered:
        xmin, ymin, xmax, ymax = box[:4]
        obj_region[ymin:ymax, xmin:xmax] = 1
        # pix_change_obj += np.sum(mask[ymin:ymax, xmin:xmax])
    for box in bboxes:
        xmin, ymin, xmax, ymax = box[:4]
        obj_region[ymin:ymax, xmin:xmax] = 1
        # pix_change_obj += np.sum(mask[ymin:ymax, xmin:xmax])
    pix_change_obj += np.sum(mask * obj_region)
    pix_change = np.sum(mask)
    pix_change_bg = pix_change - pix_change_obj

    return pix_change, (mask*255).astype(np.uint8), pix_change_obj, pix_change_bg


def object_pixel_difference(old_frame, new_frame, boxes, thresh=35):
    '''
    Compute the number of boxed pixel value differences which are greater than
    thresh
    '''
    # thresh = 35 is used in Glimpse paper
    pdb.set_trace()
    diff = np.absolute(new_frame - old_frame)
    box_diff_list = []
    for box in boxes:
        # TODO: check the box
        x_min = box[0]
        x_max = box[0] + box[2]
        # x_max =  box[2]
        y_min = box[1]
        y_max = box[1] + box[3]
        # y_max = box[3]
        box_diff_list.append(np.sum(np.greater(diff[y_min:y_max, x_min:x_max],
                                               thresh)))
    return np.sum(box_diff_list)


def compute_target_frame_rate(frame_rate_list, f1_list, target_f1=0.9):
    '''
    compute target frame rate when target f1 is achieved
    '''
    index = frame_rate_list.index(max(frame_rate_list))
    f1_list_normalized = [x/f1_list[index] for x in f1_list]
    result = [(y, x) for x, y in sorted(zip(f1_list_normalized,
                                            frame_rate_list))]
    # print(list(zip(frame_rate_list,f1_list_normalized)))
    frame_rate_list_sorted = [x for (x, _) in result]
    f1_list_sorted = [y for (_, y) in result]
    index = next(x[0] for x in enumerate(f1_list_sorted) if x[1] > target_f1)
    if index == 0:
        target_frame_rate = frame_rate_list_sorted[0]
        return target_frame_rate, -1, f1_list_sorted[index], -1, frame_rate_list_sorted[index]
    else:
        point_a = (f1_list_sorted[index-1], frame_rate_list_sorted[index-1])
        point_b = (f1_list_sorted[index], frame_rate_list_sorted[index])

        target_frame_rate = interpolation(point_a, point_b, target_f1)
        return target_frame_rate, f1_list_sorted[index-1], f1_list_sorted[index], \
               frame_rate_list_sorted[index-1], frame_rate_list_sorted[index]


# write glimpse detection results to file
def write_pipeline_result(frame_start, frame_end, gt_annot, dt_glimpse,
                          frame_flag, csvf):
    for i in range(frame_start, frame_end + 1):
        csvf.write(str(i) + ',')
        gt_boxes_final = gt_annot[i].copy()
        dt_boxes_final = dt_glimpse[i].copy()

        gt_string = []
        for box in gt_boxes_final:
            gt_string.append(' '.join([str(x) for x in box]))

        dt_string = []
        for box in dt_boxes_final:
            dt_string.append(' '.join([str(x) for x in box]))

        csvf.write(';'.join(gt_string) + ',')
        csvf.write(';'.join(dt_string) + ',' + str(frame_flag[i]) + '\n')
    csvf.close()
    return


def findDistance(r1, c1, r2, c2):
    d = (r1-r2)**2 + (c1-c2)**2
    d = d**0.5
    return d


def tracking_boxes(vis, oldFrameGray, newFrameGray, new_frame_id, old_boxes,
                   tracking_error_thresh, image_resolution):
    '''
    tracking the bboxes in oldFrameGray and return new bboxes tracked by
    optical flow.
    '''
    # find corners
    lk_params = dict(winSize=(15, 15),
                     maxLevel=5,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    feature_params = dict(maxCorners=50,
                          qualityLevel=0.01,
                          minDistance=7,
                          blockSize=7)

    mask = np.zeros_like(oldFrameGray)

    old_corners = []
    for x, y, xmax, ymax, t, score, obj_id in old_boxes:
        mask[y:ymax, x:xmax] = 255
        corners = cv2.goodFeaturesToTrack(oldFrameGray[y:ymax, x:xmax],
                                          **feature_params)
        if corners is not None:
            corners[:, 0, 0] = corners[:, 0, 0] + x
            corners[:, 0, 1] = corners[:, 0, 1] + y
            old_corners.append(corners)
    if not old_corners:
        # old_corners = np.concatenate(old_corners)
        return True, [], None, None
        # cv2.imshow(str('bad'), oldFrameGray)
        # cv2.moveWindow(str('bad'), 0, 0)
        # cv2.waitKey(0)
        # cv2.destroyWindow(str('bad'))
    else:
        old_corners = np.concatenate(old_corners)

    # old_corners = cv2.goodFeaturesToTrack(oldFrameGray, mask=mask,
    #                                       **feature_params)
    # old_corners = cv2.goodFeaturesToTrack(oldFrameGray, 26, 0.01, 7,
    #                                       mask=mask)
    new_corners, st, err = cv2.calcOpticalFlowPyrLK(oldFrameGray, newFrameGray,
                                                    old_corners, None,
                                                    **lk_params)
    # old_corners_r, st, err = cv2.calcOpticalFlowPyrLK(newFrameGray,
    #                                                   oldFrameGray,
    #                                                   old_corners, None,
    #                                                   **lk_params)
    # d = abs(old_corners-old_corners_r).reshape(-1, 2).max(-1)
    # good = d < 1
    # pdb.set_trace()
    # new_corners_copy = new_corners.copy()
    new_corners = new_corners[st == 1].reshape(-1, 1, 2)
    old_corners = old_corners[st == 1].reshape(-1, 1, 2)
    # new_corners = new_corners[good]
    # old_corners = old_corners[good]

    for new_c, old_c in zip(new_corners, old_corners):
        cv2.circle(vis, (new_c[0][0], new_c[0][1]), 5, (0, 255, 255), -1)
        cv2.circle(vis, (old_c[0][0], old_c[0][1]), 5, (0, 0, 0), -1)

    new_boxes = []

    for x, y, xmax, ymax, t, score, obj_id in old_boxes:
        indices = []
        for idx, (old_c, new_c) in enumerate(zip(old_corners, new_corners)):
            # print(old_corner)
            # cv2.circle(newFrameGray, (old_corner[0][0], old_corner[0][1]),
            #            5, (255, 0, 0))
            if old_c[0][0] >= x and old_c[0][0] <= xmax and \
               old_c[0][1] >= y and old_c[0][1] <= ymax:
                indices.append(idx)
        if not indices:
            debug_print('frame {}: object {} disappear'.format(new_frame_id,
                                                               obj_id))
            continue
        indices = np.array(indices)

        old_bbox = cv2.boundingRect(old_corners[indices])
        new_bbox = cv2.boundingRect(new_corners[indices])
        cv2.rectangle(vis, (old_bbox[0], old_bbox[1]),
                      (old_bbox[0]+old_bbox[2], old_bbox[1]+old_bbox[3]),
                      (0, 0, 0), 3)
        cv2.rectangle(vis, (new_bbox[0], new_bbox[1]),
                      (new_bbox[0]+new_bbox[2], new_bbox[1]+new_bbox[3]),
                      (0, 255, 255), 3)

        # TODO: corner cleaning

        # checking tracking error threshold condition
        displacement_vectors = []
        dist_list = []
        for old_corner, new_corner in zip(old_corners[indices],
                                          new_corners[indices]):
            dist_list.append(np.linalg.norm(new_corner-old_corner))
            displacement_vectors.append(new_corner-old_corner)
        tracking_err = np.std(dist_list)
        if tracking_err > tracking_error_thresh:
            # tracking failure, this is a trigger frame
            debug_print('frame {}: '
                        'object {} std {} > tracking error thresh {}, '
                        'tracking fails'.format(new_frame_id, obj_id,
                                                np.std(dist_list),
                                                tracking_error_thresh))
            return False, [], obj_id, tracking_err

        # update bouding box translational movement
        mean_displacement_vector = np.mean(displacement_vectors, axis=0)
        # print(mean_displacement_vector)

        # update bouding box zooming movement
        old_dists = []
        new_dists = []
        old_x_lengths = []
        old_y_lengths = []
        new_x_lengths = []
        new_y_lengths = []
        for i in range(len(old_corners[indices])):
            for j in range(i+1, len(old_corners[indices])):
                c_i = old_corners[indices][i]
                c_j = old_corners[indices][j]
                dist = np.linalg.norm(c_j-c_i)
                old_dists.append(dist)
                old_x_lengths.append(c_j[0][0]-c_i[0][0])
                old_y_lengths.append(c_j[0][1]-c_i[0][1])

                c_i = new_corners[indices][i]
                c_j = new_corners[indices][j]
                dist = np.linalg.norm(c_j-c_i)
                new_dists.append(dist)
                new_x_lengths.append(c_j[0][0]-c_i[0][0])
                new_y_lengths.append(c_j[0][1]-c_i[0][1])

        min_x_idx = np.argmin(old_corners[indices, 0, 0])
        max_x_idx = np.argmax(old_corners[indices, 0, 0])
        min_y_idx = np.argmin(old_corners[indices, 0, 1])
        max_y_idx = np.argmax(old_corners[indices, 0, 1])
        # pdb.set_trace()
        old_x_len = old_corners[indices, 0, 0][max_x_idx] - \
            old_corners[indices, 0, 0][min_x_idx]
        old_y_len = old_corners[indices, 0, 1][max_y_idx] - \
            old_corners[indices, 0, 1][min_y_idx]

        new_x_len = new_corners[indices, 0, 0][max_x_idx] - \
            new_corners[indices, 0, 0][min_x_idx]
        new_y_len = new_corners[indices, 0, 1][max_y_idx] - \
            new_corners[indices, 0, 1][min_y_idx]
        # print(old_corners[min_x_idx, 0, 0], old_corners[min_y_idx, 0, 0])

        # ratio = np.nanmean(np.array(new_dists)/np.array(old_dists))
        w_ratio = new_x_len/old_x_len
        h_ratio = new_y_len/old_y_len
        # use bbox to compute ratio
        w_ratio = new_bbox[2]/old_bbox[2]
        h_ratio = new_bbox[3]/old_bbox[3]

        if np.isnan(w_ratio):
            w_ratio = 1
        if np.isnan(h_ratio):
            h_ratio = 1

        old_center_x = (x + xmax)/2
        old_center_y = (y + ymax)/2
        new_center_x = old_center_x + mean_displacement_vector[0][0]
        new_center_y = old_center_y + mean_displacement_vector[0][1]
        # use bbox to compute displacement
        # new_center_x = old_center_x + new_bbox[0]-old_bbox[0]
        # new_center_y = old_center_y + new_bbox[1]-old_bbox[1]
        new_w = (xmax-x) * w_ratio
        new_h = (ymax-y) * h_ratio
        # new_w = ((xmax-x) + new_w)/2
        # new_h = ((ymax-y) + new_h)/2

        new_x = int(new_center_x - new_w/2)
        new_y = int(new_center_y - new_h/2)

        new_xmax = int(new_x + int(new_w))
        new_ymax = int(new_y + int(new_h))

        # new_x = int(x+mean_displacement_vector[0][0])
        # new_y = int(y+mean_displacement_vector[0][1])
        # # new_xmax = xmax
        # # new_ymax = ymax
        # new_xmax = int(new_x + (xmax - x) * ratio)
        # new_ymax = int(new_y + (ymax - y) * ratio)
        new_boxes.append([new_x, new_y, new_xmax, new_ymax, t, score, obj_id])

    return True, new_boxes, None, None


def eval_pipeline_accuracy(frame_start, frame_end,
                           gt_annot, dt_glimpse, iou_thresh=0.5):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    gt_cn = 0
    dt_cn = 0
    for i in range(frame_start, frame_end + 1):
        gt_boxes_final = gt_annot[i].copy()
        dt_boxes_final = dt_glimpse[i].copy()
        gt_cn += len(gt_boxes_final)
        dt_cn += len(dt_boxes_final)
        tp[i], fp[i], fn[i] = eval_single_image(gt_boxes_final, dt_boxes_final,
                                                iou_thresh)

    tp_total = sum(tp.values())
    fn_total = sum(fn.values())
    fp_total = sum(fp.values())

    return compute_f1(tp_total, fp_total, fn_total)


def object_appearance(start, end, gt):
    '''
    Takes start frame, end frame, and groundtruth
    Returns two dicts.
    1. object to frame range
    2. frame id to new object id
    '''
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


def object_duration_analysis(obj_to_frame_range):
    duration = list()
    for obj_id in obj_to_frame_range:
        frame_range = obj_to_frame_range[obj_id]
        duration.append(frame_range[1] - frame_range[0])
    plot_cdf(duration, 10000, 'duration', 'duration', 'duration')
    plt.show()


def bbox_to_ids(bboxes):
    return [str(box[-1]) for box in bboxes]


def visualize(newFrameGray, mask, gt, dt_glimpse, i, frame_to_new_obj):

    # print(newFrameGray.shape)
    for [x, y, xmax, ymax, t, score, obj_id] in filter_frame_detections(dt_glimpse[i], target_types={3, 8}):
        try:
            cv2.rectangle(newFrameGray, (x, y), (xmax, ymax),
                          (255, 255, 255), 3)
        except TypeError:
            pdb.set_trace()

    for [x, y, xmax, ymax, t, score, obj_id] in filter_frame_detections(gt[i], target_types={3, 8}):
        cv2.rectangle(newFrameGray, (x, y), (xmax, ymax), (0, 0, 0), 1)
        if frame_to_new_obj is not None and i in frame_to_new_obj and \
                obj_id in frame_to_new_obj[i]:
            cv2.putText(newFrameGray, 'new', (x-10, y-10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

    img_2_show = np.hstack((newFrameGray, mask))
    # cv2.imshow(str(i), img_2_show)
    cv2.imwrite('vis/{}.jpg'.format(i), img_2_show)
    return True
    # cv2.moveWindow(str(i), 0, 0)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     return False
    # else:
    #     # cv2.destroyWindow(str(i))
    #     return True


def pipeline_perfect_tracking(img_path, gt, frame_start, frame_end,
                              frame_difference_thresh, tracking_error_thresh,
                              img_name_format, view=False, mask_flag=False):
    '''
    Only frame difference triggering is implemented in this function.
    When frame difference triggers a frame, boxes are fetched from ground
    truth. Otherwise, the objects tracked by previous frame will have their
    boxes in this frame fetched from groundtruth. Therefore, tracking is
    perfect.
    '''

    filtered_gt = filter_video_detections(gt, target_types={3, 8})
    obj_to_frame_range, frame_to_new_obj = object_appearance(frame_start,
                                                             frame_end,
                                                             filtered_gt)
    # object_duration_analysis(obj_to_frame_range)
    # return
    ideally_triggered_frames = set()
    for obj_id in obj_to_frame_range:
        frame_range = obj_to_frame_range[obj_id]
        ideally_triggered_frames.add(frame_range[0])
    ideal_nb_triggered = len(ideally_triggered_frames)

    # frame_flag = {}
    dt_glimpse = defaultdict(list)
    triggered_frame = 0
    cn = 0
    video_trigger_log = list()

    # start from the first frame
    # The first frame has to be sent to server.
    img_name = img_path + img_name_format.format(frame_start)
    oldFrameGray = cv2.imread(img_name, 0)
    assert oldFrameGray is not None, 'cannot read image {}'.format(img_name)
    lastTriggeredFrameGray = oldFrameGray.copy()
    last_triggered_frame_idx = frame_start

    # get detection from server
    dt_glimpse[frame_start] = gt[frame_start]
    # if detection is obtained from server, set frame_flag to 1
    # frame_flag[frame_start] = 1
    pix_change = np.zeros_like(oldFrameGray)
    if view:
        view = visualize(oldFrameGray, pix_change, gt, dt_glimpse, frame_start,
                         frame_to_new_obj)
    frame_trigger_log = {'frame id': frame_start, 'triggered': 1,
                         'frame difference': -1,
                         'objects': bbox_to_ids(dt_glimpse[frame_start]),
                         'threshold': float(frame_difference_thresh)}
    video_trigger_log.append(frame_trigger_log)
    triggered_frame += 1  # count triggered fames
    cn += 1
    tp = 0
    # run the pipeline for the rest of the frames
    for i in range(frame_start+1, frame_end+1):

        img_name = img_path + img_name_format.format(i)
        newFrameGray = cv2.imread(img_name, 0)
        assert newFrameGray is not None, 'cannot read {}'.format(img_name)

        pix_change = np.zeros_like(newFrameGray)
        # mask out bboxes
        if mask_flag:
            mask = np.ones_like(newFrameGray)
            # print('last trigger frame {} has boxes:'
            #       .format(last_triggered_frame_idx))
            for box in dt_glimpse[last_triggered_frame_idx]:
                # print('\t', box)
                xmin, ymin, xmax, ymax, t = box[:5]
                if t not in {3, 8}:
                    mask[ymin:ymax, xmin:xmax] = 0
            # print('previous frame {} has boxes:'.format(i-1))

            # mask off the non-viechle objects in current frame
            for box in gt[i]:
                # print('\t', box)
                xmin, ymin, xmax, ymax, t = box[:5]
                if t not in {3, 8}:
                    mask[ymin:ymax, xmin:xmax] = 0
            # for box in dt_glimpse[i-1]:
            #     # print('\t', box)
            #     xmin, ymin, xmax, ymax, t = box[:5]
            #     if t not in [3, 8]:
            #         mask[ymin:ymax, xmin:xmax] = 0
            lastTriggeredFrameGray_masked = lastTriggeredFrameGray.copy()*mask
            newFrameGray_masked = newFrameGray.copy() * mask
            # compute frame difference
            frame_diff, pix_change = frame_difference(
                    lastTriggeredFrameGray_masked, newFrameGray_masked)
        else:
            # compute frame difference
            frame_diff, pix_change = frame_difference(
                    lastTriggeredFrameGray, newFrameGray)
        if frame_diff > frame_difference_thresh:
            # triggered
            # run inference to get the detection results
            dt_glimpse[i] = gt[i].copy()
            triggered_frame += 1
            oldFrameGray = newFrameGray.copy()
            lastTriggeredFrameGray = oldFrameGray.copy()
            debug_print('frame diff {} > th {}, trigger {}, last trigger {}'
                        .format(frame_diff, frame_difference_thresh, i,
                                last_triggered_frame_idx))
            last_triggered_frame_idx = i
            frame_trigger_log = {'frame id': i, 'triggered': 1,
                                 'frame difference': float(frame_diff),
                                 'objects': bbox_to_ids(dt_glimpse[i]),
                                 'threshold': float(frame_difference_thresh)}
            video_trigger_log.append(frame_trigger_log)
            if i in ideally_triggered_frames:
                # print('frame diff {} > th {}, trigger frame {}, tp'
                #       .format(frame_diff, frame_difference_thresh, i))
                tp += 1
                ideally_triggered_frames.remove(i)
            elif i-1 in ideally_triggered_frames:
                tp += 1
                ideally_triggered_frames.remove(i-1)
            elif i-2 in ideally_triggered_frames:
                tp += 1
                ideally_triggered_frames.remove(i-2)
            elif i-3 in ideally_triggered_frames:
                tp += 1
                ideally_triggered_frames.remove(i-3)
            else:
                # print('frame diff {} > th {}, trigger extra frame {}, fp'
                #       .format(frame_diff, frame_difference_thresh, i))
                pass
            if view:
                view = visualize(newFrameGray, pix_change, gt, dt_glimpse, i,
                                 frame_to_new_obj)
                # mask *= 255
                # view = visualize(newFrameGray, mask, gt, dt_glimpse, i,
                #                  frame_to_new_obj)

        else:
            if view:
                view = visualize(newFrameGray, pix_change, gt, dt_glimpse,
                                 i, frame_to_new_obj)
                # mask *= 255
                # view = visualize(newFrameGray, mask, gt, dt_glimpse,
                #                  i, frame_to_new_obj)
            if i in ideally_triggered_frames:
                # print('frame diff {} < th {}, miss triggering {}, fn'
                #       .format(frame_diff, frame_difference_thresh, i))
                # view = visualize(newFrameGray, pix_change, gt, dt_glimpse,
                #                  i, frame_to_new_obj)
                pass

            # for box in dt_glimpse[i-1]:
            #     print(len(box))

            obj_id_in_perv_frame = [box[6] for box in dt_glimpse[i-1]]
            # assume perfect tracking, the boxes of all objects in previously
            # detected frame will be get the boxes in current frame from the
            # groundtruth
            dt_in_cur_frame = list()
            for box in gt[i]:
                if box[-1] in obj_id_in_perv_frame:
                    dt_in_cur_frame.append(box)
            dt_glimpse[i] = dt_in_cur_frame
            frame_trigger_log = {'frame id': i, 'triggered': 0,
                                 'frame difference': float(frame_diff),
                                 'objects': bbox_to_ids(dt_glimpse[i]),
                                 'threshold': float(frame_difference_thresh)}
            video_trigger_log.append(frame_trigger_log)
    cv2.destroyAllWindows()

    # write_pipeline_result(frame_start, frame_end, gt_annot, dt_glimpse,
    #                         frame_flag, csvf)
    filtered_dt_glimpse = filter_video_detections(dt_glimpse,
                                                  target_types={3, 8})
    f1 = eval_pipeline_accuracy(frame_start, frame_end, filtered_gt,
                                filtered_dt_glimpse)
    fp = triggered_frame - tp
    fn = len(ideally_triggered_frames)
    print('tp={}, fp={}, fn={}, nb_triggered={}, nb_ideal={}'
          .format(tp, fp, fn, triggered_frame, ideal_nb_triggered))

    trigger_f1 = compute_f1(tp, fp, fn)
    return triggered_frame, ideal_nb_triggered, f1, trigger_f1, \
        video_trigger_log


def pipeline(img_path, gt, frame_start, frame_end, image_resolution,
             frame_diff_th, tracking_err_th, img_name_format,
             display_flag=False):
    '''
    This is an end to end pipeline of glimpse with frame difference triggering
    and tracking
    '''
    # display_flag = True

    obj_to_frame_range, frame_to_new_obj = object_appearance(frame_start,
                                                             frame_end, gt)
    # object_duration_analysis(obj_to_frame_range)
    # return
    ideally_triggered_frames = set()
    for obj_id in obj_to_frame_range:
        frame_range = obj_to_frame_range[obj_id]
        ideally_triggered_frames.add(frame_range[0])
    ideal_nb_triggered = len(ideally_triggered_frames)
    frames_log = list()
    dt_glimpse = defaultdict(list)
    triggered_frame = 0
    last_triggered_frame_gray = None
    prev_frame_gray = None
    tp = 0

    # start from the first frame
    # The first frame has to be sent to server.
    # get detection from server
    # if detection is obtained from server, set frame_flag to 1
    # run the pipeline for the rest of the frames
    for i in range(frame_start, frame_end + 1):
        frame_log = dict()
        img_name = img_path + img_name_format.format(i)
        frame = cv2.imread(img_name)
        assert frame is not None, 'cannot read image {}'.format(img_name)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if i == frame_start:
            triggered_frame += 1  # count triggered fames
            last_triggered_frame_gray = frame_gray.copy()
            dt_glimpse[i] = gt[i].copy()
            frame_log = {'frame id': i,
                         'initial trigger': 1,
                         'frame differece trigger': -1,
                         'frame difference': 0,
                         'detections': dt_glimpse[i],
                         'tracking failure trigger': 0,
                         'tracking failure object id': 'NA',
                         'tracking error': 0.0}
        else:
            # compute frame difference
            assert last_triggered_frame_gray is not None, \
                'last triggered frame is None'
            frame_diff = frame_difference(last_triggered_frame_gray,
                                          frame_gray)
            if frame_diff > frame_diff_th:
                debug_print('frame {}: frame diff {} > th {}, trigger, tp'
                            .format(i, frame_diff, frame_diff_th))
                # triggered
                # run inference to get the detection results
                dt_glimpse[i] = gt[i].copy()
                triggered_frame += 1
                last_triggered_frame_gray = frame_gray.copy()
                frame_log = {'frame id': i,
                             'initial trigger': 0,
                             'frame differece trigger': 1,
                             'frame difference': float(frame_diff),
                             'detections': dt_glimpse[i],
                             'tracking failure trigger': 0,
                             'tracking failure object id': 'NA',
                             'tracking error': 0.0}
                if i in ideally_triggered_frames:
                    # print('frame diff {} > th {}, trigger frame {}, tp'
                    #       .format(frame_diff, frame_difference_thresh, i))
                    tp += 1
                    ideally_triggered_frames.remove(i)
                elif i-1 in ideally_triggered_frames:
                    tp += 1
                    ideally_triggered_frames.remove(i-1)
                elif i-2 in ideally_triggered_frames:
                    tp += 1
                    ideally_triggered_frames.remove(i-2)
                elif i-3 in ideally_triggered_frames:
                    tp += 1
                    ideally_triggered_frames.remove(i-3)
                else:
                    pass
            # if i in ideally_triggered_frames:
            #     # triggered and run inference to get the detection results
            #     dt_glimpse[i] = gt[i].copy()
            #     triggered_frame += 1
            #     last_triggered_frame_gray = frame_gray.copy()
            else:
                # use tracking to get the result
                assert i-1 in dt_glimpse, \
                    '{} not in detected frames'.format(i-1)
                if not dt_glimpse[i-1]:
                    # prev frame is empty, then current frame is empty
                    dt_glimpse[i] = []
                    frame_log = {'frame id': i,
                                 'initial trigger': 0,
                                 'frame differece trigger': 0,
                                 'frame difference': -1,
                                 'detections': dt_glimpse[i],
                                 'tracking failure trigger': 0,
                                 'tracking failure object id': 'NA',
                                 'tracking error': 0.0}
                else:
                    # prev frame is not empty, do tracking
                    status, new_boxes, failure_obj_id, failure_tracking_err = \
                        tracking_boxes(frame, prev_frame_gray, frame_gray, i,
                                       dt_glimpse[i-1], tracking_err_th,
                                       image_resolution)
                    if status:
                        dt_glimpse[i] = new_boxes
                        frame_log = {'frame id': i,
                                     'initial trigger': 0,
                                     'frame differece trigger': 0,
                                     'frame difference': -1,
                                     'detections': dt_glimpse[i],
                                     'tracking failure trigger': 0,
                                     'tracking failure object id': 'NA',
                                     'tracking error': 0.0}
                    else:
                        # tracking failed and trigger a new frame
                        dt_glimpse[i] = gt[i].copy()
                        triggered_frame += 1
                        last_triggered_frame_gray = frame_gray.copy()
                        frame_log = {'frame id': i,
                                     'initial trigger': 0,
                                     'frame differece trigger': 0,
                                     'frame difference': -1,
                                     'detections': dt_glimpse[i],
                                     'tracking failure trigger': 1,
                                     'tracking failure object id':
                                     str(failure_obj_id),
                                     'tracking error':
                                     float(failure_tracking_err)}
        prev_frame_gray = frame_gray.copy()
        frames_log.append(frame_log)
        if display_flag:
            display_flag = visualize(frame, gt, dt_glimpse, i,
                                     frame_to_new_obj)

    f1 = eval_pipeline_accuracy(frame_start, frame_end, gt, dt_glimpse)
    fp = triggered_frame - tp
    fn = len(ideally_triggered_frames)
    debug_print('tp={}, fp={}, fn={}, nb_triggered={}, nb_ideal={}'
                .format(tp, fp, fn, triggered_frame, ideal_nb_triggered))

    trigger_f1 = compute_f1(tp, fp, fn)
    return triggered_frame, ideal_nb_triggered, f1, trigger_f1, frames_log


def pipeline_frame_select(img_path, gt, frame_start, frame_end,
                              frame_difference_thresh, tracking_error_thresh,
                              img_name_format, view=False, mask_flag=False):
    '''
    Only frame difference triggering is implemented in this function.
    Tracking module is not implmented. Instead, detections from the previsous
    frames are used so that this pipeline is at the same comparison level as
    videostorm.
    '''

    filtered_gt = filter_video_detections(gt, height_range=(720//20, 720), target_types={3, 8})
    obj_to_frame_range, frame_to_new_obj = object_appearance(frame_start,
                                                             frame_end,
                                                             filtered_gt)
    # object_duration_analysis(obj_to_frame_range)
    # return
    ideally_triggered_frames = set()
    for obj_id in obj_to_frame_range:
        frame_range = obj_to_frame_range[obj_id]
        ideally_triggered_frames.add(frame_range[0])
    ideal_nb_triggered = len(ideally_triggered_frames)

    # frame_flag = {}
    dt_glimpse = defaultdict(list)
    triggered_frame = 0
    cn = 0
    video_trigger_log = list()

    # start from the first frame
    # The first frame has to be sent to server.
    img_name = img_path + img_name_format.format(frame_start)
    oldFrameGray = cv2.imread(img_name, 0)
    assert oldFrameGray is not None, 'cannot read image {}'.format(img_name)
    lastTriggeredFrameGray = oldFrameGray.copy()
    last_triggered_frame_idx = frame_start

    # get detection from server
    dt_glimpse[frame_start] = gt[frame_start]
    # if detection is obtained from server, set frame_flag to 1
    # frame_flag[frame_start] = 1
    pix_change = np.zeros_like(oldFrameGray)
    if view:
        view = visualize(oldFrameGray, pix_change, gt, dt_glimpse, frame_start,
                         frame_to_new_obj)
    frame_trigger_log = {'frame id': frame_start, 'triggered': 1,
                         'frame difference': -1,
                         'objects': bbox_to_ids(dt_glimpse[frame_start]),
                         'threshold': float(frame_difference_thresh)}
    video_trigger_log.append(frame_trigger_log)
    triggered_frame += 1  # count triggered fames
    cn += 1
    tp = 0

    pix_change_obj_list = list()
    pix_change_bg_list = list()

    # run the pipeline for the rest of the frames
    for i in range(frame_start+1, frame_end+1):

        img_name = img_path + img_name_format.format(i)
        newFrameGray = cv2.imread(img_name, 0)
        assert newFrameGray is not None, 'cannot read {}'.format(img_name)

        pix_change = np.zeros_like(newFrameGray)
        # mask out bboxes
        if mask_flag:
            mask = np.ones_like(newFrameGray)
            # print('last trigger frame {} has boxes:'
            #       .format(last_triggered_frame_idx))
            for box in dt_glimpse[last_triggered_frame_idx]:
                # print('\t', box)
                xmin, ymin, xmax, ymax, t = box[:5]
                if t not in {3, 8}:
                    mask[ymin:ymax, xmin:xmax] = 0
            # print('previous frame {} has boxes:'.format(i-1))

            # mask off the non-viechle objects in current frame
            for box in gt[i]:
                # print('\t', box)
                xmin, ymin, xmax, ymax, t = box[:5]
                if t not in {3, 8}:
                    mask[ymin:ymax, xmin:xmax] = 0
            # for box in dt_glimpse[i-1]:
            #     # print('\t', box)
            #     xmin, ymin, xmax, ymax, t = box[:5]
            #     if t not in [3, 8]:
            #         mask[ymin:ymax, xmin:xmax] = 0
            lastTriggeredFrameGray_masked = lastTriggeredFrameGray.copy()*mask
            newFrameGray_masked = newFrameGray.copy() * mask
            # compute frame difference
            frame_diff, pix_change, pix_change_obj, pix_change_bg = frame_difference(
                    lastTriggeredFrameGray_masked, newFrameGray_masked,
                    filter_frame_detections(dt_glimpse[last_triggered_frame_idx],
                                            target_types={3,8}, height_range=(720//20, 720)),
                    filter_frame_detections(gt[i],
                                            target_types={3,8}, height_range=(720//20, 720)))
            pix_change_obj_list.append(pix_change_obj)
            pix_change_bg_list.append(pix_change_bg)
        else:
            # compute frame difference
            frame_diff, pix_change, pix_change_obj, pix_change_bg = frame_difference(
                    lastTriggeredFrameGray, newFrameGray, dt_glimpse[last_triggered_frame_idx],
                    filter_frame_detections(dt_glimpse[last_triggered_frame_idx],
                                            target_types={3,8}, height_range=(720//20, 720)),
                    filter_frame_detections(gt[i],
                                            target_types={3,8}, height_range=(720//20, 720)))
        if frame_diff > frame_difference_thresh:
            # triggered
            # run inference to get the detection results
            dt_glimpse[i] = gt[i].copy()
            triggered_frame += 1
            oldFrameGray = newFrameGray.copy()
            lastTriggeredFrameGray = oldFrameGray.copy()
            debug_print('frame diff {} > th {}, trigger {}, last trigger {}'
                        .format(frame_diff, frame_difference_thresh, i,
                                last_triggered_frame_idx))
            last_triggered_frame_idx = i
            frame_trigger_log = {'frame id': i, 'triggered': 1,
                                 'frame difference': float(frame_diff),
                                 'objects': bbox_to_ids(dt_glimpse[i]),
                                 'threshold': float(frame_difference_thresh)}
            video_trigger_log.append(frame_trigger_log)
            if i in ideally_triggered_frames:
                # print('frame diff {} > th {}, trigger frame {}, tp'
                #       .format(frame_diff, frame_difference_thresh, i))
                tp += 1
                ideally_triggered_frames.remove(i)
            elif i-1 in ideally_triggered_frames:
                tp += 1
                ideally_triggered_frames.remove(i-1)
            elif i-2 in ideally_triggered_frames:
                tp += 1
                ideally_triggered_frames.remove(i-2)
            elif i-3 in ideally_triggered_frames:
                tp += 1
                ideally_triggered_frames.remove(i-3)
            else:
                # print('frame diff {} > th {}, trigger extra frame {}, fp'
                #       .format(frame_diff, frame_difference_thresh, i))
                pass
            if view:
                view = visualize(newFrameGray, pix_change, gt, dt_glimpse, i,
                                 frame_to_new_obj)
                # mask *= 255
                # view = visualize(newFrameGray, mask, gt, dt_glimpse, i,
                #                  frame_to_new_obj)

        else:
            if view:
                view = visualize(newFrameGray, pix_change, gt, dt_glimpse,
                                 i, frame_to_new_obj)
                # mask *= 255
                # view = visualize(newFrameGray, mask, gt, dt_glimpse,
                #                  i, frame_to_new_obj)
            if i in ideally_triggered_frames:
                # print('frame diff {} < th {}, miss triggering {}, fn'
                #       .format(frame_diff, frame_difference_thresh, i))
                # view = visualize(newFrameGray, pix_change, gt, dt_glimpse,
                #                  i, frame_to_new_obj)
                pass

            dt_glimpse[i] = dt_glimpse[i-1]
            frame_trigger_log = {'frame id': i, 'triggered': 0,
                                 'frame difference': float(frame_diff),
                                 'objects': bbox_to_ids(dt_glimpse[i]),
                                 'threshold': float(frame_difference_thresh)}
            video_trigger_log.append(frame_trigger_log)
        # print('gt frame {}: {}'.format(i, filtered_gt[i]))
        # print('dt frame {}: {}'.format(i, filter_frame_detections(dt_glimpse[i], height_range=(720//20, 720), target_types={3,8})))
    cv2.destroyAllWindows()

    # write_pipeline_result(frame_start, frame_end, gt_annot, dt_glimpse,
    #                         frame_flag, csvf)
    filtered_dt_glimpse = filter_video_detections(dt_glimpse, height_range=(720//20, 720),
                                                  target_types={3, 8})
    f1 = eval_pipeline_accuracy(frame_start, frame_end, filtered_gt,
                                filtered_dt_glimpse)
    fp = triggered_frame - tp
    fn = len(ideally_triggered_frames)
    print('tp={}, fp={}, fn={}, nb_triggered={}, nb_ideal={}, pix_change_obj={}, pix_change_bg={}'
          .format(tp, fp, fn, triggered_frame, ideal_nb_triggered, np.mean(pix_change_obj_list), np.mean(pix_change_bg_list)))

    trigger_f1 = compute_f1(tp, fp, fn)
    return triggered_frame, ideal_nb_triggered, f1, trigger_f1, \
        video_trigger_log, np.mean(pix_change_obj_list), np.mean(pix_change_bg_list)


# def pipeline_perfect_trigger(img_path, dt_annot, gt_annot,
#                              frame_start, frame_end, image_resolution,
#                              frame_difference_thresh,
#                              tracking_error_thresh, kitti=True):
#     '''
#     Only tracking is implemented in this function. Frame difference
#     triggering is replaced by perfect triggring when a new object shows up.
#     '''
#     view = True
#     # view = False
#     obj_to_frame_range, frame_to_new_obj = object_appearance(frame_start,
#                                                              frame_end,
#                                                              gt_annot)
#     # object_duration_analysis(obj_to_frame_range)
#     # return
#     ideally_triggered_frames = set()
#     for obj_id in obj_to_frame_range:
#         frame_range = obj_to_frame_range[obj_id]
#         ideally_triggered_frames.add(frame_range[0])
#     ideal_nb_triggered = len(ideally_triggered_frames)
#     # print('at least trigger', len(ideally_triggered_frames))
#
#     frame_flag = {}
#     dt_glimpse = defaultdict(list)
#     triggered_frame = 0
#     cn = 0

#     # start from the first frame
#     # The first frame has to be sent to server.
#     if kitti:
#         img_name = format(frame_start, '010d') + '.png'
#     else:
#         img_name = format(frame_start, '06d') + '.jpg'

#     oldFrameGray = cv2.imread(img_path + img_name, 0)
#     # lastTriggeredFrameGray = oldFrameGray.copy()
#     # get detection from server
#     dt_glimpse[frame_start] = dt_annot[frame_start]
#     # if detection is obtained from server, set frame_flag to 1
#     frame_flag[frame_start] = 1
#     triggered_frame += 1  # count triggered fames
#     cn += 1
#     last_index = frame_start
#     # run the pipeline for the rest of the frames
#     for i in range(frame_start + 1, frame_end + 1):
#         if kitti:
#             newImgName = img_path + format(i, '010d') + '.png'
#         else:
#             newImgName = img_path + format(i, '06d') + '.jpg'
#         newFrameGray = cv2.imread(newImgName, 0)

#         # No need to compute frame differece
#         # Use ideal trigger
#         if i in ideally_triggered_frames:
#             # triggered
#             # run inference to get the detection results
#             dt_glimpse[i] = dt_annot[i].copy()
#             triggered_frame += 1
#             oldFrameGray = newFrameGray.copy()
#             # lastTriggeredFrameGray = oldFrameGray.copy()
#             frame_flag[i] = 1
#         else:
#             # use tracking to get the result
#             assert last_index in dt_glimpse, print(last_index)
#             if not dt_glimpse[last_index]:
#                 # last frame is empty, then current frame is empty
#                 dt_glimpse[i] = []
#                 oldFrameGray = newFrameGray.copy()
#                 frame_flag[i] = 0
#             else:
#                 # need to use tracking to get detection results
#                 # track from last frame
#                 for [x, y, xmax, ymax, t, score, obj_id] in \
#                         dt_glimpse[last_index]:  # dt_annot[i - 1]:
#                     # print(i, x, y, xmax-x, ymax-y, obj_id)
#                     roi = oldFrameGray[y:ymax, x:xmax]
#                     # find corners
#                     old_corners = cv2.goodFeaturesToTrack(roi, 26, 0.01, 7)

#                     # add 4 corners of the bounding box as the feature points
#                     if old_corners is not None:
#                         old_corners[:, 0, 0] = old_corners[:, 0, 0] + x
#                         old_corners[:, 0, 1] = old_corners[:, 0, 1] + y
#                         # old_corners = np.append(old_corners,
#                         #                         [[[np.float32(x),
#                         #                            np.float32(y)]]],
#                         #                         axis=0)
#                         # old_corners = np.append(old_corners,
#                         #                         [[[np.float32(xmax),
#                         #                            np.float32(y)]]],
#                         #                         axis=0)
#                         # old_corners = np.append(old_corners,
#                         #                         [[[np.float32(x),
#                         #                            np.float32(ymax)]]],
#                         #                         axis=0)
#                         # old_corners = np.append(old_corners,
#                         #                         [[[np.float32(xmax),
#                         #                            np.float32(ymax)]]],
#                         #                         axis=0)
#                     else:
#                         continue
#                         old_corners = np.array([[[np.float32(x),
#                                                   np.float32(y)]]])
#                         old_corners = np.append(old_corners,
#                                                 [[[np.float32(xmax),
#                                                    np.float32(y)]]],
#                                                 axis=0)
#                         old_corners = np.append(old_corners,
#                                                 [[[np.float32(x),
#                                                    np.float32(ymax)]]],
#                                                 axis=0)
#                         old_corners = np.append(old_corners,
#                                                 [[[np.float32(xmax),
#                                                    np.float32(ymax)]]],
#                                                 axis=0)
#                     # print(len(old_corners), old_corners[0])

#                     # No need to read tracking code
#                     # print(type(old_corners))
#                     try:
#                         ft_pts_box = cv2.boundingRect(old_corners)
#                     except TypeError:
#                         print(type(old_corners))
#                         assert False
#                     [newX, newY, newW, newH] = tracking(oldFrameGray,
#                                                         newFrameGray,
#                                                         old_corners,
#                                                         tracking_error_thresh,
#                                                         image_resolution,
#                                                         ymax-y)
#                     if not (newX+newY+newW+newH):  # track failure
#                         track_success_flag = 0
#                         break
#                     elif newW < 15 or newH < 15:
#                         # object too small = it disappears
#                         track_success_flag = 1
#                     else:
#                         # add tracking result as glimpse result
#                         # New Method
#                         old_center = [ft_pts_box[0] + ft_pts_box[2]//2,
#                                       ft_pts_box[1] + ft_pts_box[3]//2]
#                         new_center = [newX+newW//2, newY+newH//2]
#                         displacement = [new_center[0] - old_center[0],
#                                         new_center[1] - old_center[1]]
#                         w_ratio = 1  # newW/ft_pts_box[2]
#                         h_ratio = 1  # newH/ft_pts_box[3]
#                         w = int((xmax-x) * w_ratio)
#                         h = int((ymax-y) * h_ratio)
#                         x += displacement[0]
#                         y += displacement[1]
#                         # print(w, h)
#                         dt_glimpse[i].append([x, y, x+w, y+h, t,
#                                               score, obj_id])
#                         # Old Method
#                         # dt_glimpse[i].append([newX, newY, newW+newX,
#                         #                       newH+newY, t, score, obj_id])
#                         track_success_flag = 1
#                 if track_success_flag == 0:
#                     # tracking fails, get detection result from server
#                     dt_glimpse[i] = dt_annot[i].copy()
#                     triggered_frame += 1
#                     frame_flag[i] = 1
#                     # lastTriggeredFrameGray = oldFrameGray.copy()
#                 else:
#                     if i not in dt_glimpse:
#                         dt_glimpse[i] = []
#                     frame_flag[i] = 0
#                 oldFrameGray = newFrameGray.copy()

#         last_index = i
#         # print(i, frame_flag[i], len(dt_glimpse[i]))
#         # if [] in dt_glimpse[i]:
#         #   continue
#         if view:
#             for [x, y, xmax, ymax, t, score, obj_id] in dt_glimpse[i]:
#                 cv2.rectangle(newFrameGray, (x, y), (xmax, ymax),
#                               (0, 255, 0), 3)
#                 cv2.putText(newFrameGray, 'dt', (x-10, y-10),
#                             cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
#             for [x, y, xmax, ymax, t, score, obj_id] in gt_annot[i]:
#                 cv2.rectangle(newFrameGray, (x, y), (xmax, ymax),
#                               (255, 255, 0), 3)
#             try:
#                 cv2.imshow(str(i) + '_' + str(frame_flag[i]), newFrameGray)
#                 cv2.waitKey(0)
#                 cv2.destroyWindow(str(i) + '_' + str(frame_flag[i]))
#             except KeyboardInterrupt:
#                 # break
#                 view = False
#         # pdb.set_trace()
#     cv2.destroyAllWindows()

#     print("triggered frames", triggered_frame)

#     # write_pipeline_result(frame_start, frame_end, gt_annot,
#     #                       dt_glimpse, frame_flag, csvf)
#     f1 = eval_pipeline_accuracy(frame_start, frame_end, gt_annot, dt_glimpse)

#     return triggered_frame, ideal_nb_triggered, f1

# This is the old implementation of tracking which follows the glimpse paper.
# For now, this version of tracking does not work well, so a better
# implementation is adopted.

# def tracking(oldFrameGray, newFrameGray, old_corners, tracking_error_thresh,
#              image_resolution, h):
#     lk_params = dict(maxLevel=2)
#     new_corners, st, err = cv2.calcOpticalFlowPyrLK(oldFrameGray,
#                                                     newFrameGray,
#                                                     old_corners, None,
#                                                     **lk_params)
#     new_corners = new_corners[st == 1].reshape(-1, 1, 2)
#     old_corners = old_corners[st == 1].reshape(-1, 1, 2)
#     if len(new_corners) < 4:
#         print('No enough feature points')
#         return [0, 0, 0, 0]  # this object disappears

#     r_add, c_add = 0, 0
#     for corner in new_corners:
#         r_add = r_add + corner[0][1]
#         c_add = c_add + corner[0][0]
#     centroid_row = int(1.0*r_add/len(new_corners))
#     centroid_col = int(1.0*c_add/len(new_corners))
#     # draw centroid
#     cv2.circle(newFrameGray, (int(centroid_col), int(centroid_row)), 5,
#                (255, 0, 0))
#     # add only those corners to new_corners_updated
#     # which are at a distance of 30 or lesse

#     new_corners_updated = new_corners.copy()
#     old_corners_updated = old_corners.copy()

#     tobedel = []
#     dist_list = []

#     for index in range(len(new_corners)):
#         # remove coners that are outside the image
#         if new_corners[index][0][0] > image_resolution[0] or \
#            new_corners[index][0][0] < 0 or \
#            new_corners[index][0][1] > image_resolution[1] or \
#            new_corners[index][0][1] < 0:
#             tobedel.append(index)

#         # remove outliers
#         if findDistance(new_corners[index][0][1], new_corners[index][0][0],
#                         int(centroid_row), int(centroid_col)) > 2*h:
#             tobedel.append(index)

#         dist = np.linalg.norm(new_corners[index] - old_corners[index])
#         dist_list.append(dist)

#     # TODO: uncomment this
#     dist_median = np.median(dist_list)
#     dist_std = np.std(dist_list)

#     for index in range(len(new_corners)):
#         if dist_list[index] > dist_median + 3*dist_std or \
#            dist_list[index] < dist_median - 3*dist_std:
#             tobedel.append(index)

#     new_corners_updated = np.delete(new_corners_updated, tobedel, 0)
#     old_corners_updated = np.delete(old_corners_updated, tobedel, 0)

#     # if there are not enough feature points, then assume this object
#     # disappears in current frame
#     if len(new_corners_updated) < 4:
#         # print('No enough feature points')
#         return [0, 0, 1, 1]  # this object disappears
#     # (x_bound, y_bound, w_bound, h_bound) = \
#     #     cv2.boundingRect(new_corners_updated)
#     # cv2.rectangle(newFrameGray, (x_bound, y_bound),
#     #               (x_bound + w_bound, y_bound + h_bound), (255, 0, 0), 1)

#     x_list = []
#     y_list = []
#     for corner in new_corners_updated:
#         cv2.circle(newFrameGray, (int(corner[0][0]), int(corner[0][1])),
#                    5, (0, 255, 0))
#         x_list.append(int(corner[0][0]))
#         y_list.append(int(corner[0][1]))

#     dist_list = []
#     for index in range(len(new_corners_updated)):
#         dist = np.linalg.norm(new_corners_updated[index] -
#                               old_corners_updated[index])
#         dist_list.append(dist)
#     # print([(x,y) for (x,y) in zip(new_corners_updated,
#                                     old_corners_updated)])
#     # print(np.std(dist_list))
#     if np.std(dist_list) > tracking_error_thresh:
#         # tracking failure, this is a trigger frame
#         print('std {} > tracking error thresh {}, tracking fails'
#               .format(np.std(dist_list), tracking_error_thresh))
#         return [0, 0, 0, 0]  # indicates tracking failure
#     else:
#         x = min(x_list)
#         y = min(y_list)
#         w = max(x_list) - x
#         h = max(y_list) - y
#         return [x, y, w, h]
#     # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
