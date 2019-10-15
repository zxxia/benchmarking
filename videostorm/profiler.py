""" VideoStrom Profiler """
import copy
from collections import defaultdict
from utils.model_utils import eval_single_image
from utils.utils import interpolation, compute_f1


def profile(video_name, gt, full_model_dt, start_frame, end_frame, frame_rate,
            temporal_sampling_list, f_profile, target_f1=0.9):
    """ profile a list of frame rate """
    f1_score_list = []
    for sample_rate in temporal_sampling_list:
        tpos = defaultdict(int)
        fpos = defaultdict(int)
        fneg = defaultdict(int)
        save_dt = []

        for img_index in range(start_frame, end_frame+1):
            dt_boxes_final = []
            current_full_model_dt = full_model_dt[img_index]
            current_gt = gt[img_index]
            # based on sample rate, decide whether this frame is sampled
            if img_index % sample_rate >= 1:
                # this frame is not sampled, so reuse the last saved
                # detection result
                dt_boxes_final = copy.deepcopy(save_dt)
            else:
                # this frame is sampled, so use the full model result
                dt_boxes_final = copy.deepcopy(current_full_model_dt)
                save_dt = copy.deepcopy(dt_boxes_final)

            tpos[img_index], fpos[img_index], fneg[img_index] = \
                eval_single_image(current_gt, dt_boxes_final)

        tp_total = sum(tpos.values())
        fp_total = sum(fpos.values())
        fn_total = sum(fneg.values())

        f1_score = compute_f1(tp_total, fp_total, fn_total)
        print('relative fps={}, f1={}'.format(1/sample_rate, f1_score))
        f1_score_list.append(f1_score)
        f_profile.write(','.join([video_name, str(1/sample_rate),
                                  str(f1_score)])+'\n')

    frame_rate_list = [frame_rate/x for x in temporal_sampling_list]

    current_f1_list = f1_score_list

    if current_f1_list[-1] < target_f1:
        target_frame_rate = None
        target_frame_rate = frame_rate
    else:
        index = next(x[0] for x in enumerate(current_f1_list)
                     if x[1] > target_f1)
        if index == 0:
            target_frame_rate = frame_rate_list[0]
        else:
            point_a = (current_f1_list[index-1], frame_rate_list[index-1])
            point_b = (current_f1_list[index], frame_rate_list[index])
            target_frame_rate = interpolation(point_a, point_b, target_f1)

    # select best profile
    smallest_gpu_time = 100*frame_rate
    gpu_time = 100*target_frame_rate

    if gpu_time <= smallest_gpu_time:
        best_frame_rate = target_frame_rate

    return best_frame_rate


def profile_eval(gt, full_model_dt, best_sample_rate, start_frame, end_frame):
    """ evaluation """
    tpos = defaultdict(int)
    fpos = defaultdict(int)
    fneg = defaultdict(int)
    save_dt = []

    for img_index in range(start_frame, end_frame + 1):
        dt_boxes_final = []
        current_full_model_dt = full_model_dt[img_index]
        current_gt = gt[img_index]
        # based on sample rate, decide whether this frame is sampled
        if img_index % best_sample_rate >= 1:
            # this frame is not sampled, so reuse the last saved
            # detection result
            dt_boxes_final = copy.deepcopy(save_dt)

        else:
            # this frame is sampled, so use the full model result
            dt_boxes_final = copy.deepcopy(current_full_model_dt)
            save_dt = copy.deepcopy(dt_boxes_final)

        tpos[img_index], fpos[img_index], fneg[img_index] = \
            eval_single_image(current_gt, dt_boxes_final)

    tp_total = sum(tpos.values())
    fp_total = sum(fpos.values())
    fn_total = sum(fneg.values())

    f1_score = compute_f1(tp_total, fp_total, fn_total)
    return f1_score
