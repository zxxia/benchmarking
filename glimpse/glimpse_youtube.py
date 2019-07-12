import cv2
import time
import numpy as np
from collections import defaultdict
import os
from glimpse_kitti import pipeline, compute_target_frame_rate
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt

kitti = False

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
    target_f1 = 0.9
  
    frame_rate_dict = {'traffic': 30,
                       'highway_no_traffic': 25,
                       'highway_normal_traffic': 30,
                       'street_racing': 30,
                       'reckless_driving': 30,
                       'motor': 24,
                      }
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
  
    # standard_frame_rate = 10.0 # for comparison with KITTI
    # path = '/home/zhujun/video_analytics_pipelines/dataset/Youtube/'
    # path = '/Users/zhujunxiao/Desktop/benchmarking/New_Dataset/'
    path = '/home/zxxia/videos/'
  
    # result file: save the frame rate needed for Glimpse 
    final_result_f = open('youtube_glimpse_result_COCO.csv','w')
  
    videos = ['traffic', 'highway_no_traffic', 'highway_normal_traffic', 'reckless_driving', 'motor']
    #'traffic', 'highway_no_traffic', 
    # videos = ['highway_no_traffic']
    for video_type in videos:
        image_resolution = image_resolution_dict[video_type]
        frame_rate = frame_rate_dict[video_type]
        
        # Two parameters are used for Glimpse, one (para1) is frame difference thresh
        # The other is tracking error thresh (para2)
        para1_list = [0.1, 1, 2, 5, 10] #[1,2,5,10]#
        para2_list = [0.1, 1, 2 ,5, 10]
        # read ground truth and full model detection result
        # image name, detection result, ground truth
        gt_filename = 'updated_gt_FasterRCNN_COCO.csv'
        annot_path = path + video_type + '/profile/' + gt_filename
        img_path = path + video_type +'/'
        # read full model detection results and ground truth 
        gt_annot, dt_annot, frame_end = get_gt_dt(annot_path)
        # segment the video into 30s    
        chunk_length = 30 # 30 seconds
        num_seg = frame_end // (frame_rate * chunk_length) 
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
            
  
            for para1 in para1_list:
                for para2 in para2_list:
                    detail_f.write(str(seg_index) + ',' + str(para1) + ',' + str(para2) + ',')
  
                    dt_glimpse_file = result_direct + str(seg_index) + '_' + str(para1) + '_'+ str(para2) + '.csv'
                    csvf = open(dt_glimpse_file, 'w')
                    print(para1, para2)
                    # larger para1, smaller thresh, easier to be triggered
                    frame_difference_thresh = image_resolution[0]*image_resolution[1]/para1 
                    tracking_error_thresh = para2
                    # images start from index 1
                    start = seg_index * (frame_rate * chunk_length) + 1 
                    end = (seg_index + 1) * (frame_rate * chunk_length)
                    print(img_path)
                    triggered_frame, f1 = pipeline(img_path, dt_annot, gt_annot, start, 
                                                   end, csvf, image_resolution, 
                                                   frame_rate, frame_difference_thresh, 
                                                   tracking_error_thresh, False)
  
                    current_frame_rate = triggered_frame / float(chunk_length)
                    frame_rate_list.append(current_frame_rate)
                    f1_list.append(f1)
                    detail_f.write(str(current_frame_rate) + ',' + str(f1) + '\n')
                    print("frame rate, f1 score:", current_frame_rate, f1)
                    label_str = "{0:.2f}, {1:.2f}".format(para1, para2)
                    print(label_str)
                    
                    # result_f.write(str(current_frame_rate)+','+str(f1)+','+'\n')  
  
            f1_list.append(1.0)
            frame_rate_list.append(frame_rate)
            target_frame_rate = compute_target_frame_rate(frame_rate_list, f1_list, target_f1)
   
  
            final_result_f.write(video_type + '_' + str(seg_index) + ',' + str(target_frame_rate/frame_rate) + '\n')
            print(target_frame_rate)
        detail_f.close()
        
        #fig = plt.figure() 
        relative_frame_rate_list = [fr/frame_rate for fr in frame_rate_list]
        plt.scatter(f1_list, relative_frame_rate_list) #, label=label_str)
        plt.xlabel('target f1 score')
        plt.ylabel('minimum frame rate needed')
        plt.title(video_type + " & target frame rate="+str(target_frame_rate))
        #plt.close(fig)
        plt.legend(loc='upper left')
        #plt.show()
        plt.savefig("/home/zxxia/figs/glimpse/"+video_type+".png")
  
    final_result_f.close()

if __name__=='__main__':
    main()
