import cv2
import time
import numpy as np
from collections import defaultdict
import os
from glimpse_kitti import pipeline, compute_target_frame_rate
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
					if t == 3:
					# if h > 40:
						gt_annot[frame_id].append([x,y,w,h,t])
						dt_annot[frame_id].append([x,y,w,h,t])

	return gt_annot, dt_annot, frame_end


def main():
	target_f1 = 0.9
	frame_rate_dict = {'walking': 30,
					   'driving_downtown': 30, 
					   'highway': 25, 
					   'crossroad2': 30,
					   'crossroad': 30,
				   	   'crossroad3': 30,
					   'crossroad4': 30,
					   'crossroad5': 30,
					   'driving1': 30,
					   'driving2': 30}
	image_resolution_dict = {'walking': [3840,2160],
							 'driving_downtown': [3840, 2160], 
							 'highway': [1280,720],
							 'crossroad2': [1920,1080],
							 'crossroad': [1920,1080],
							 'crossroad3': [1280,720],
							 'crossroad4': [1920,1080],
							 'crossroad5': [1920,1080],
							 'driving1': [1920,1080],
							 'driving2': [1280,720]
							 }

	chunk_length = 30 # 30 seconds
	# standard_frame_rate = 10.0 # for comparison with KITTI
	# path = '/home/zhujun/video_analytics_pipelines/dataset/Youtube/'
	path = '/Users/zhujunxiao/Desktop/benchmarking/New_Dataset/'

	inference_time = 100 # avg. GPU processing  time
	final_result_f = open('youtube_glimpse_result_COCO.csv','w')

	for video_type in ['crossroad2']:#,'crossroad','crossroad2','crossroad5','driving_downtown','walking']:#image_resolution_dict.keys():
		image_resolution = image_resolution_dict[video_type]
		frame_rate = frame_rate_dict[video_type]
		standard_frame_rate = frame_rate
		para1_list = [1,1.2,1.3,1.4,1.5,2,2.2,2.3,2.5,4,4.5,5] #[1,2,5,10]#
		para2_list = [2]#[0.5,10,50]
		# resize the currsent video frames to 10Hz
		resize_rate = frame_rate/standard_frame_rate 
		# read ground truth and full model detection result
		# image name, detection result, ground truth
		gt_filename = 'updated_gt_FasterRCNN_COCO.csv'
		# gt_filename = 'updated_input_w_gt.csv'
		annot_path = path + video_type + '/profile/' + gt_filename

		# annot_path = path + video_type + '/updated_input_w_gt.csv'
		img_path = path + video_type +'/images/'
		gt_annot, dt_annot, frame_end = get_gt_dt(annot_path)
		num_seg = frame_end // (frame_rate * chunk_length) # segment the video into 30s
		detail_file = annot_path.replace(gt_filename, 'glimpse_result_0612.csv')
		detail_f = open(detail_file, 'w')
		detail_f.write('seg_index, frame difference factor,'\
			'tracking error thresh, avg Frame rate, f1 score\n')

		for seg_index in range(num_seg):
			print(video_type, seg_index)
			# Run inference on the first frame
			# Two parameters: frame difference threshold, tracking error thresh
			result_direct = annot_path.replace(gt_filename,
											   'glimpse_result_0612/')
			if not os.path.exists(result_direct):
				os.makedirs(result_direct)

			frame_rate_list = []
			f1_list = []
			for para1 in para1_list:
				for para2 in para2_list:
					detail_f.write(str(seg_index) + ',' + str(para1) + ',' + 
								   str(para2) + ',')

					dt_glimpse_file = result_direct + str(seg_index) + '_' + \
									str(para1) + '_'+ str(para2) + '.csv'
					csvf = open(dt_glimpse_file, 'w')
					print(para1, para2)
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
					detail_f.write(str(current_frame_rate) + ',' + str(f1) + '\n')
					print("frame rate, f1 score:", current_frame_rate, f1)

					# result_f.write(str(current_frame_rate)+','+str(f1)+','+'\n')	

			f1_list.append(1.0)
			frame_rate_list.append(standard_frame_rate)
			target_frame_rate = compute_target_frame_rate(frame_rate_list,
														  f1_list,
														  target_f1)

			# plt.scatter(f1_list, frame_rate_list)
			# plt.show()

			final_result_f.write(video_type + '_' + str(seg_index) + ',' + 
								 str(target_frame_rate/standard_frame_rate) + '\n')
			print(target_frame_rate)
		detail_f.close()
	final_result_f.close()


	return











if __name__=='__main__':
	main()