
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from my_utils import IoU, interpolation

def load_full_model_detection(fullmodel_detection_path, height):
    full_model_dt = {}
    with open(fullmodel_detection_path, 'r') as f:
        for line in f:
            line_list = line.strip().split(',')
            # real image index starts from 1
            img_index = int(line_list[0].split('.')[0]) #- 1
            if not line_list[1]: # no detected object
                gt_boxes_final = []
            else:
                gt_boxes_final = []
                gt_boxes = line_list[1].split(';')
                for gt_box in gt_boxes:
                    # t is object type
                    tmp = [int(i) for i in gt_box.split(' ')]
                    assert len(tmp) == 6, print(tmp, line)
                    x = tmp[0]
                    y = tmp[1]
                    w = tmp[2]
                    h = tmp[3]
                    t = tmp[4]
                    if t == 3 or t == 8: # choose car and truch objects
                        # if h > height/float(20): # ignore objects that are too small
                        gt_boxes_final.append([x, y, x+w, y+h, t])
            full_model_dt[img_index] = gt_boxes_final
            
    return full_model_dt, img_index

def eval_single_image_single_type(gt_boxes, pred_boxes, iou_thresh):
    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = IoU(pred_box, gt_box)
            if iou > iou_thresh:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    return tp, fp, fn

def eval_single_image(gt_boxes, dt_boxes, iou_thresh=0.5):
    tp_dict = {}
    fp_dict = {}
    fn_dict = {}
    gt = defaultdict(list)  
    dt = defaultdict(list)
    for box in gt_boxes:
        gt[box[4]].append(box[0:4])
    for box in dt_boxes:
        dt[box[4]].append(box[0:4])

    for t in gt.keys():
        current_gt = gt[t]
        current_dt = dt[t]
        tp_dict[t], fp_dict[t], fn_dict[t] = eval_single_image_single_type(
                                             current_gt, current_dt, iou_thresh)

    tp = sum(tp_dict.values())
    fp = sum(fp_dict.values())
    fn = sum(fn_dict.values())
    extra_t = [t for t in dt.keys() if t not in gt]
    for t in extra_t:
        fp += len(dt[t])
    # print(tp, fp, fn)
    return tp, fp, fn




def main():
    iou_thresh = 0.5
    dataset_list = ['traffic', 'highway_no_traffic', 
    'highway_normal_traffic', 'reckless_driving', 'motor']
    color_dict =  {
    'traffic' : 'r', 
    'highway_no_traffic' : 'g', 
    'highway_normal_traffic' : 'blue', 
    'reckless_driving' : 'black', 
    'motor' : 'yellow',
    }
    frame_rate_dict = {'walking': 30,
                       'driving_downtown': 30, 
                       'highway': 25, 
                       'crossroad2': 30,
                       'crossroad': 30,
                       'crossroad3': 30,
                       'crossroad4': 30,
                       'crossroad5': 30,
                       'driving1': 30,
                       'driving2': 30,
                       'traffic': 30,
                       'highway_no_traffic': 25,
                       'highway_normal_traffic': 30,
                       'street_racing': 30,
                       'reckless_driving': 30,
                       'motor': 24}
    image_resolution_dict = {'walking': [3840,2160],
                             'driving_downtown': [3840, 2160], 
                             'highway': [1280,720],
                             'crossroad2': [1920,1080],
                             'crossroad': [1920,1080],
                             'crossroad3': [1280,720],
                             'crossroad4': [1920,1080],
                             'crossroad5': [1920,1080],
                             'driving1': [1920,1080],
                             'driving2': [1280,720],
                             'traffic': [1280,720],
                             'highway_no_traffic': [1280,720],
                             'highway_normal_traffic': [1280,720],
                             'street_racing': [1280,720],
                             'motor': [1280,720],
                             'reckless_driving': [1280,720],
                             }


    # VideoStorm result file
    fileID = open('VideoStorm_result_tmp.csv', 'w')
    # path where video and ground truth file is saved 
    data_path = '/home/zxxia/videos/'
    # assume we have a target f1 score, then we compute what is the cost (GPU 
    # processing time)
    target_f1 = 0.9
    chunk_length = 30 # chunk a long video into 30-second short videos
    # Change sampling rate to change video frame rate
    temporal_sampling_list = [20,15,10,5,4,3,2.5,2,1.8,1.5,1.2,1]


    for video_name in dataset_list:
        # run the full model on each frame first, and use it as ground truth
        frame_rate = frame_rate_dict[video_name] 
        height = image_resolution_dict[video_name][1]
        # load ground truth
        fullmodel_detection_path = data_path + video_name + '/profile/updated_gt_FasterRCNN_COCO.csv'
        print(fullmodel_detection_path)

        full_model_dt, num_of_frames = load_full_model_detection(
                                                fullmodel_detection_path, height)
        F1_score_list = defaultdict(list)


        for sample_rate in temporal_sampling_list:
            # tp = true positive 
            # fp = false positive
            # fn = false negative
            # first for each frame, compute tp, fp, fn
            tp = defaultdict(int)
            fp = defaultdict(int)
            fn = defaultdict(int)
            # if current frame is not sampled, use the detection results of 
            # previous sampled frame 
            save_dt = []

            for img_index in range(0, num_of_frames):
                dt_boxes_final = []
                current_full_model_dt = full_model_dt[img_index]

                # based on sample rate, decide whether this frame is sampled
                if img_index%sample_rate >= 1:
                    # this frame is not sampled, so reuse the last saved
                    # detection result
                    dt_boxes_final = [box for box in save_dt]

                else:
                    # this frame is sampled, so use the full model result
                    dt_boxes_final = [box for box in current_full_model_dt]
                    save_dt = [box for box in dt_boxes_final]

                # for each frame, compute tp, fp, fn
                # you don't need to read eval_single_image, just use it
                tp[img_index], fp[img_index], fn[img_index] = \
                    eval_single_image(current_full_model_dt, dt_boxes_final)
            

            tp_total = defaultdict(int)
            fp_total = defaultdict(int)
            fn_total = defaultdict(int)
            # compute total tp, fp, fn for each short video (chunk)
            for index in range(num_of_frames):
                key = index // int(chunk_length*frame_rate)

                tp_total[key] += tp[index]
                fn_total[key] += fn[index]
                fp_total[key] += fp[index]


            for key in tp_total.keys():
                # print(fn_total[key] + tp_total[key])
                if tp_total[key]:
                    precison = float(tp_total[key]) / (tp_total[key] + fp_total[key])
                    recall = float(tp_total[key]) / (tp_total[key] + fn_total[key])
                    f1 = 2*(precison*recall)/(precison+recall)
                else:
                    if fn_total[key]:
                        f1 = 0
                    else:
                        f1 = 1 

                F1_score_list[key].append(f1)


        # cost depends on frame rate. The higher the frame rate, the longer GPU
        # time is needed.
        frame_rate_list = [frame_rate/x for x in temporal_sampling_list]
        relative_frame_rate_list = [fr/frame_rate_dict[video_name] for fr in frame_rate_list]
        # print(F1_score_list.keys())
        # print(len(frame_rate_list))
        for key in sorted(F1_score_list.keys()):
            current_f1_list = F1_score_list[key]
            # print(list(zip(frame_rate_list, current_f1_list)))

            if current_f1_list[-1] == 0:
                target_frame_rate = frame_rate
            else:
                # compute target frame rate using interpolation
                F1_score_norm = [x/current_f1_list[-1] for x in current_f1_list]
                index = next(x[0] for x in enumerate(F1_score_norm) if x[1] > target_f1)
                if index == 0:
                    target_frame_rate = frame_rate_list[0]
                else:
                    point_a = (current_f1_list[index-1], frame_rate_list[index-1])
                    point_b = (current_f1_list[index], frame_rate_list[index])


                    target_frame_rate  = interpolation(point_a, point_b, target_f1)
            print(key, target_frame_rate)
            fileID.write(video_name+'_'+str(key)+','+str(target_frame_rate)+'\n')

            #plt.plot(relative_frame_rate_list, F1_score_norm,'-o', color=color_dict[video_name], label=video_name)
    #       detail_f.write(video_name+'_'+str(key)+','
    #           +' '.join(str(x) for x in frame_rate_list)+','
    #           +' '.join([str(x) for x in F1_score_list[key]])+'\n')
    # plt.legend(loc="lower right")
    # plt.xlabel("Frame rate")
    # plt.ylabel("F1_score")
    # plt.title("Accurarcy vs Frame Rate")
    # plt.show()


if __name__ == '__main__':
  main()
