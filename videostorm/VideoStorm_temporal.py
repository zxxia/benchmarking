
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from utils.model_utils import load_full_model_detection, eval_single_image
from utils.utils import IoU, interpolation, load_metadata, compute_f1

# path where video and ground truth file is saved 
data_path = '/mnt/data/zhujun/new_video/'
# assume we have a target f1 score, then we compute what is the cost (GPU 
# processing time)
target_f1 = 0.9
chunk_length = 30 # chunk a long video into 30-second short videos
# Change sampling rate to change video frame rate
temporal_sampling_list = [20,15,10,5,4,3,2.5,2,1.8,1.5,1.2,1]
iou_thresh = 0.5
dataset_list = sorted(['traffic', 'reckless_driving','motor', 'jp_hw', 'russia', 
                 'tw_road', 'tw_under_bridge', 'highway_normal_traffic', 
                 'highway_no_traffic', 'tw', 'tw1', 'jp', 'russia1', 'drift', 
                 'park', 'nyc', 'lane_split', 'walking',  'highway', 'crossroad2', 
                 'crossroad', 'crossroad3', 'crossroad4', 'driving1', 'driving2',])


# color_dict =  {
# 'traffic' : 'r', ]
# 'highway_no_traffic' : 'g', 
# 'highway_normal_traffic' : 'blue', 
# 'reckless_driving' : 'black', 
# 'motor' : 'yellow',
# }


def main():
    # VideoStorm result file
    
    with open('videostorm_result.csv', 'w') as fileID:
        fileID.write('video name,relative frame rate\n')
        for video_name in dataset_list:
            metadata = load_metadata(data_path + video_name + '/metadata.json')
            # run the full model on each frame first, and use it as ground truth
            frame_rate = metadata['frame rate'] 
            height = metadata['resolution'][1]
            # load ground truth
            fullmodel_detection_path = data_path + video_name + '/profile/updated_gt_FasterRCNN_COCO.csv'
            print(fullmodel_detection_path)

            full_model_dt, num_of_frames = load_full_model_detection(fullmodel_detection_path, height)
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

                for img_index in range(1, num_of_frames+1):
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
                for index in range(1,num_of_frames+1):
                    key = index // (int(chunk_length*frame_rate)+1)
                    if key >= num_of_frames // int(chunk_length * frame_rate):
                        break

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
            relative_frame_rate_list = [fr/frame_rate for fr in frame_rate_list]
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

                print(video_name, key, target_frame_rate)
                fileID.write(video_name+'_'+str(key)+','+str(target_frame_rate/frame_rate)+'\n')

                # plt.plot(relative_frame_rate_list, F1_score_norm,'-o', label=video_name)
              # detail_f.write(video_name+'_'+str(key)+','
              #     +' '.join(str(x) for x in frame_rate_list)+','
              #     +' '.join([str(x) for x in F1_score_list[key]])+'\n')
    # plt.legend(loc="lower right")
    # plt.xlabel("Frame rate")
    # plt.ylabel("F1_score")
    # plt.title("Accurarcy vs Frame Rate")
    # plt.show()


if __name__ == '__main__':
  main()
