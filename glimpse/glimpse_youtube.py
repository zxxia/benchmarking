import cv2
import time
import numpy as np
from collections import defaultdict
import os
from glimpse_kitti import pipeline, compute_target_frame_rate
import matplotlib
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from my_utils import load_metadata

kitti = False
TARGET_F1 = 0.9
# segment the video into 30s    
CHUNK_LENGTH = 30 # 30 seconds
PATH = '/mnt/data/zhujun/new_video/'
VIDEOS = sorted(['traffic', 'reckless_driving','motor', 'jp_hw', 'russia', 
                 'tw_road', 'tw_under_bridge', 'highway_normal_traffic', 
                 'highway_no_traffic', 'tw', 'tw1', 'jp', 'russia1','drift', 
                 'park'])

# Two parameters are used for Glimpse, one (para1) is frame difference thresh
# The other is tracking error thresh (para2)
PARA1_LIST = [2,4,6,8,10] #[1,2,5,10]#
PARA2_LIST = [0.5,1,3,5,7,10] #[0.1, 1, 2 ,5, 10]

def get_gt_dt(annot_path):
    # read ground truth and full model (server side) detection results
    gt_annot = defaultdict(list)
    dt_annot = defaultdict(list)
    frame_end = 0
    with open(annot_path, 'r') as f:
        for line in f:
            annot_list = line.strip().split(',')
            frame_id = int(annot_list[0].replace('.jpg','')) 
            frame_end = frame_id
  
            gt_str = annot_list[1] # use full model detection results as ground truth
            gt_boxes = gt_str.split(';')
            if gt_boxes == ['']:
                gt_annot[frame_id] = []
            else:
                for box in gt_boxes:
                    box_list = box.split(' ')
                    x = int(box_list[0])
                    y = int(box_list[1])
                    w = int(box_list[2])
                    h = int(box_list[3])
                    t = int(box_list[4])
                    if t == 3 or t == 8: # only select car and truck
                    # if h > 40:
                        gt_annot[frame_id].append([x,y,w,h,t])
                        dt_annot[frame_id].append([x,y,w,h,t])
  
    return gt_annot, dt_annot, frame_end


def main():
    # result file: save the frame rate needed for Glimpse 
    final_result_f = open('glimpse_result.csv','w')
  
    for video_type in VIDEOS:
        metadata = load_metadata(PATH + video_type + '/metadata.json')

        image_resolution = metadata['resolution']
        frame_rate = metadata['frame rate']
        
        # read ground truth and full model detection result
        # image name, detection result, ground truth
        gt_filename = 'updated_gt_FasterRCNN_COCO.csv'
        annot_path = PATH + video_type + '/profile/' + gt_filename
        img_path = PATH + video_type +'/'
        # read full model detection results and ground truth 
        gt_annot, dt_annot, frame_end = get_gt_dt(annot_path)
        num_seg = frame_end // (frame_rate * CHUNK_LENGTH) 
        # save the results (glimpse detection results) to detail file
        detail_file = annot_path.replace(gt_filename, 'glimpse_result_0612.csv')
        detail_f = open(detail_file, 'w')
        detail_f.write('seg_index, frame difference factor,'\
          'tracking error thresh, avg Frame rate, f1 score\n')
        frame_rate_list = []
        f1_list = []
       
        
        for seg_index in range(num_seg):
            print(video_type, seg_index)
            # Run inference on the first frame
            # Two parameters: frame difference threshold, tracking error thresh
            result_direct = annot_path.replace(gt_filename,'glimpse_result_0612/')
            if not os.path.exists(result_direct):
                os.makedirs(result_direct)
            
  
            for para1 in PARA1_LIST:
                for para2 in PARA2_LIST:
                    detail_f.write(str(seg_index) + ',' 
                                   + str(para1) + ',' + str(para2) + ',')
  
                    dt_glimpse_file = result_direct + str(seg_index) + '_' + str(para1) + '_'+ str(para2) + '.csv'
                    csvf = open(dt_glimpse_file, 'w')
                    #print(para1, para2)
                    # larger para1, smaller thresh, easier to be triggered
                    frame_difference_thresh = image_resolution[0]*image_resolution[1]/para1 
                    tracking_error_thresh = para2
                    # images start from index 1
                    start = seg_index * (frame_rate * CHUNK_LENGTH) + 1 
                    end = (seg_index + 1) * (frame_rate * CHUNK_LENGTH)
                    print(img_path)
                    triggered_frame, f1 = pipeline(img_path, dt_annot, gt_annot,
                                                   start, end, csvf, 
                                                   image_resolution, frame_rate,
                                                   frame_difference_thresh, 
                                                   tracking_error_thresh, False)
  
                    current_frame_rate = triggered_frame / float(CHUNK_LENGTH)
                    frame_rate_list.append(current_frame_rate)
                    f1_list.append(f1)
                    detail_f.write(str(current_frame_rate) + ',' + str(f1) + '\n')
                    #print("frame rate, f1 score:", current_frame_rate, f1)
                    #label_str = "{0:.2f}, {1:.2f}".format(para1, para2)
                    #print(label_str)
                    
                    # result_f.write(str(current_frame_rate)+','+str(f1)+','+'\n')  
  
            f1_list.append(1.0)
            frame_rate_list.append(frame_rate)
            target_frame_rate = compute_target_frame_rate(frame_rate_list, 
                                                          f1_list, TARGET_F1)
   
  
            final_result_f.write(video_type + '_' 
                                 + str(seg_index) + ',' 
                                 + str(target_frame_rate/frame_rate) + '\n')
            #print(target_frame_rate)
        detail_f.close()
        
        #fig = plt.figure() 
        #relative_frame_rate_list = [fr/frame_rate for fr in frame_rate_list]
        #plt.scatter(f1_list, relative_frame_rate_list) #, label=label_str)
        #plt.xlabel('target f1 score')
        #plt.ylabel('minimum frame rate needed')
        #plt.title(video_type + " & target frame rate="+str(target_frame_rate))
        #plt.close(fig)
        #plt.legend(loc='upper left')
        #plt.show()
        #plt.savefig("/home/zxxia/figs/glimpse/"+video_type+".png")
  
    final_result_f.close()

if __name__=='__main__':
    main()
