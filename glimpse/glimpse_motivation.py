import cv2
import time
import numpy as np
from collections import defaultdict
import os
from glimpse_kitti import pipeline, compute_target_frame_rate
from matplotlib import pyplot as plt

kitti = False

def get_gt_dt(annot_path, height):
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
					if t == 3 or t == 8: # object is car, this depends on the task
						gt_annot[frame_id].append([x,y,w,h,t])
						dt_annot[frame_id].append([x,y,w,h,t])

	return gt_annot, dt_annot, frame_end


def main():
	target_f1 = 0.9
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
							 'crossroad6':[1920,1080],
							 'crossroad7':[1920,1080],
							 'cropped_crossroad3':[600,400]
							 }
	# frame_rate_dict = {'walking': 30,
	# 				   'driving_downtown': 30, 
	# 				   'highway': 25, 
	# 				   'crossroad2': 30,
	# 				   'crossroad': 30,
	# 			   	   'crossroad3': 30,
	# 				   'crossroad4': 30,
	# 				   'crossroad5': 30,
	# 				   'driving1': 30,
	# 				   'driving2': 30,
	# 				   'crossroad6':30,
	# 				   'crossroad7':30}

	# dataset_list = ['highway','crossroad','driving1','driving2','crossroad2',
	# 						'crossroad3','crossroad4','crossroad5','driving_downtown','walking']
	dataset_list = ['crossroad6']
	chunk_length = 30 # 30 seconds
	# standard_frame_rate = 10.0 # for comparison with KITTI
	path = '/home/zhujun/video_analytics_pipelines/dataset/Youtube/'
	# path = '/Users/zhujunxiao/Desktop/benchmarking/New_Dataset/'

	inference_time = 100 # avg. GPU processing  time
	# final_result_f = open('youtube_glimpse_motivation.csv','w')
	final_result_f = open('youtube_glimpse_motivation_COCO_part2_test.csv','w')

	# choose the first 3 mins to get the best frame diff thresh
	for video_type in dataset_list:
		image_resolution = image_resolution_dict[video_type]
		height = image_resolution[1]
		if 'highway' in video_type:
			frame_rate = 25
		else:
			frame_rate = 30
		# frame_rate = frame_rate_dict[video_type]
		standard_frame_rate = frame_rate
		# para1_list = [2,2.1,2.15,2.2,2.3,2.4,2.5,2.6,3] #[1,2,5,10]#
		# para2_list = [20]
		para1_list = [2,5,6,7,8,9,10,20] #[1,2,5,10]#
		para2_list = [3]
		# resize the current video frames to 10Hz
		resize_rate = 1 
		# read ground truth and full model detection result
		# image name, detection result, ground truth
		annot_path = path + video_type + '/profile/updated_gt_FasterRCNN_COCO.csv'
		img_path = path + video_type +'/'
		gt_annot, dt_annot, frame_end = get_gt_dt(annot_path, height)
		num_seg = frame_end // (frame_rate * chunk_length) # segment the video into 30s

		for seg_index in range(0, 1):
			print(video_type, seg_index)
			# Run inference on the first frame
			# Two parameters: frame difference threshold, tracking error thresh
			frame_rate_list = []
			f1_list = []
			for para1 in para1_list:
				for para2 in para2_list:
					csvf = open('no_meaning.csv','w')
					# larger para1, smaller thresh, easier to be triggered
					frame_difference_thresh = \
								image_resolution[0]*image_resolution[1]/para1	
					tracking_error_thresh = para2
					# images start from index 1
					start = seg_index * (frame_rate * chunk_length) + 1 
					end = (seg_index + 1) * (frame_rate * chunk_length)
					triggered_frame, f1 = pipeline(img_path, 
												     dt_annot, 
												     gt_annot, 
												     start, 
												     end, 
												     csvf, 
												     image_resolution,
												     frame_rate, 
												     frame_difference_thresh, 
													 tracking_error_thresh, 
													 resize_rate,
													 False)

					current_frame_rate = triggered_frame / float(chunk_length)
					frame_rate_list.append(current_frame_rate)
					f1_list.append(f1)
					print(f1, current_frame_rate)
			if max(f1_list) < target_f1:
				para1 = 20
			else:
				index = next(x[0] for x in enumerate(f1_list) 
								  if x[1] >= target_f1)
				para1 = para1_list[index]

		para2 = para2
		# use the selected parameters for the next 5 mins
		frame_difference_thresh = \
					image_resolution[0]*image_resolution[1]/para1	
		tracking_error_thresh = para2
		for seg_index in range(num_seg):
			# images start from index 1
			start = seg_index * (frame_rate * chunk_length) + 1 
			end = (seg_index + 1) * (frame_rate * chunk_length)
			csvf = open('no_meaning.csv','w')
			triggered_frame, f1 = pipeline(img_path, 
										     dt_annot, 
										     gt_annot, 
										     start, 
										     end, 
										     csvf, 
										     image_resolution,
										     frame_rate, 
										     frame_difference_thresh, 
											 tracking_error_thresh, 
											 resize_rate,
											 False)
			fps = float(triggered_frame)/float(chunk_length)
			print('F1, current_frame_rate:', f1, fps)
			final_result_f.write(video_type + '_' + str(seg_index) + ',' + str(para1) + ',' + str(f1) + ',' + str(fps) + '\n')


	return











if __name__=='__main__':
	main()