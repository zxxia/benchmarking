from collections import defaultdict
from VideoStorm_temporal import load_full_model_detection, eval_single_image
from utils.utils import interpolation, load_metadata, compute_f1

def profile(path, dataset, frame_rate, gt, start_frame, end_frame, 
            model_list, temporal_sampling_list, target_f1=0.9):
    result = {}
    # choose model
    # choose resolution
    # choose frame rate
    for model in model_list:
        F1_score_list = []
        if model == 'FasterRCNN':
            dt_file = path + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv' 
        else:
            dt_file = '/home/zhujun/video_analytics_pipelines/final_code/videostorm/small_model/' + \
                                  dataset + '_updated_gt_SSD.csv'
        full_model_dt, num_of_frames = load_full_model_detection(dt_file)
        for sample_rate in temporal_sampling_list:
            tp = defaultdict(int)
            fp = defaultdict(int)
            fn = defaultdict(int)
            save_dt = []

            for img_index in range(start_frame, end_frame + 1):
                dt_boxes_final = []
                current_full_model_dt = full_model_dt[img_index]
                current_gt = gt[img_index]
                # based on sample rate, decide whether this frame is sampled
                if img_index%sample_rate >= 1:
                    # this frame is not sampled, so reuse the last saved
                    # detection result
                    dt_boxes_final = [box for box in save_dt]

                else:
                    # this frame is sampled, so use the full model result
                    dt_boxes_final = [box for box in current_full_model_dt]
                    save_dt = [box for box in dt_boxes_final]

                tp[img_index], fp[img_index], fn[img_index] = \
                eval_single_image(current_gt, dt_boxes_final)

            tp_total = sum(tp.values())
            fp_total = sum(fp.values())
            fn_total = sum(fn.values())

            f1 = compute_f1(tp_total, fp_total, fn_total)
            F1_score_list.append(f1)

        frame_rate_list = [frame_rate/x for x in temporal_sampling_list]

        current_f1_list = F1_score_list

        if current_f1_list[-1] < target_f1:
            target_frame_rate = None
        else:
            index = next(x[0] for x in enumerate(current_f1_list) 
                                                 if x[1] > target_f1)
            if index == 0:
                target_frame_rate = frame_rate_list[0]
            else:
                point_a = (current_f1_list[index-1], frame_rate_list[index-1])
                point_b = (current_f1_list[index], frame_rate_list[index])
                target_frame_rate  = interpolation(point_a, point_b, target_f1)
        
        result[model] = target_frame_rate
        # select best profile
    good_settings = []
    smallest_gpu_time = 100*frame_rate
    for model in result.keys():
        target_frame_rate = result[model]
        if target_frame_rate == None:
            continue
        if model == 'FasterRCNN':
            gpu_time = 100*target_frame_rate
        else:
            gpu_time = 50*target_frame_rate
        
        if gpu_time < smallest_gpu_time:
            best_model = model
            best_frame_rate = target_frame_rate

    return best_model, best_frame_rate


def profile_eval(path, dataset, frame_rate, gt, best_model, best_sample_rate, 
                 start_frame, end_frame):
    if best_model == 'FasterRCNN':
        dt_file = path + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv' 
    else:
        dt_file = '/home/zhujun/video_analytics_pipelines/final_code/videostorm/small_model/' + \
                              dataset + '_updated_gt_SSD.csv'       
    full_model_dt, _ = load_full_model_detection(dt_file)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    save_dt = []

    for img_index in range(start_frame, end_frame + 1):
        dt_boxes_final = []
        current_full_model_dt = full_model_dt[img_index]
        current_gt = gt[img_index]
        # based on sample rate, decide whether this frame is sampled
        if img_index%best_sample_rate >= 1:
            # this frame is not sampled, so reuse the last saved
            # detection result
            dt_boxes_final = [box for box in save_dt]

        else:
            # this frame is sampled, so use the full model result
            dt_boxes_final = [box for box in current_full_model_dt]
            save_dt = [box for box in dt_boxes_final]

        tp[img_index], fp[img_index], fn[img_index] = \
        eval_single_image(current_gt, dt_boxes_final)   


    tp_total = sum(tp.values())
    fp_total = sum(fp.values())
    fn_total = sum(fn.values())

    f1 = compute_f1(tp_total, fp_total, fn_total)
    return f1


