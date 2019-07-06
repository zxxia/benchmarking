from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from VideoStorm_temporal import load_full_model_detection, eval_single_image
from my_utils import interpolation

path = '/home/zhujun/video_analytics_pipelines/dataset/Youtube/'
temporal_sampling_list = [20,15,10,5,4,3,2.5,2,1.8,1.5,1.2,1]
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
						 'cropped_crossroad3':[600,400],
						 'cropped_driving2':[600,400]
						 }


target_f1=0.9

def profile(dataset, frame_rate, gt, start_frame, chunk_length=30):
	result = {}
	height = image_resolution_dict[dataset][1]
	standard_frame_rate = frame_rate
	# choose model
	# model_list = ['FasterRCNN','SSD']
	model_list = ['FasterRCNN']
	# choose resolution
	# choose frame rate
	for model in model_list:
		F1_score_list = []
		if model == 'FasterRCNN':
			dt_file = path + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv' 
		else:
			dt_file = '/home/zhujun/video_analytics_pipelines/final_code/videostorm/small_model/' + \
					  dataset + '_updated_gt_SSD.csv'
		full_model_dt, num_of_frames = load_full_model_detection(dt_file, height)		
		for sample_rate in temporal_sampling_list:
			img_cn = 0
			tp = defaultdict(int)
			fp = defaultdict(int)
			fn = defaultdict(int)
			save_dt = []

			for img_index in range(start_frame, start_frame+chunk_length*frame_rate):
				dt_boxes_final = []
				current_full_model_dt = full_model_dt[img_index]
				current_gt = gt[img_index]
				resize_rate = frame_rate/standard_frame_rate
				if img_index%resize_rate >= 1:
					continue
				else:
					img_index = img_cn
					img_cn += 1


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

				# print(tp[img_index], fp[img_index],fn[img_index])
			tp_total = sum(tp.values())
			fp_total = sum(fp.values())
			fn_total = sum(fn.values())



			if tp_total:
				precison = float(tp_total) / (tp_total + fp_total)
				recall = float(tp_total) / (tp_total + fn_total)
				f1 = 2*(precison*recall)/(precison+recall)
			else:
				if fn_total:
					f1 = 0
				else:
					f1 = 1 

			F1_score_list.append(f1)

		frame_rate_list = [standard_frame_rate/x 
						for x in temporal_sampling_list]

		current_f1_list = F1_score_list
		# print(list(zip(frame_rate_list, current_f1_list)))

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
	smallest_gpu_time = 100*standard_frame_rate
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


def profile_eval(dataset, frame_rate, gt, best_model, best_sample_rate,
	start_frame, end_frame):
	height = image_resolution_dict[dataset][1]
	standard_frame_rate = frame_rate

	if best_model == 'FasterRCNN':
		dt_file = path + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv' 
	else:
		dt_file = '/home/zhujun/video_analytics_pipelines/final_code/videostorm/small_model/' + \
				  dataset + '_updated_gt_SSD.csv'	
	full_model_dt, _ = load_full_model_detection(dt_file, height)

	img_cn = 0
	tp = defaultdict(int)
	fp = defaultdict(int)
	fn = defaultdict(int)
	save_dt = []

	for img_index in range(start_frame, end_frame):
		dt_boxes_final = []
		current_full_model_dt = full_model_dt[img_index]
		current_gt = gt[img_index]
		resize_rate = frame_rate/standard_frame_rate
		if img_index%resize_rate >= 1:
			continue
		else:
			img_index = img_cn
			img_cn += 1


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
	if tp_total:
		precison = float(tp_total) / (tp_total + fp_total)
		recall = float(tp_total) / (tp_total + fn_total)
		f1 = 2*(precison*recall)/(precison+recall)
	else:
		if fn_total:
			f1 = 0
		else:
			f1 = 1 


	return f1

def main():
	iou_thresh = 0.5
	target_f1 = 0.9
	dataset_list = ['driving_downtown','highway','crossroad',
					'crossroad2','crossroad3', 'crossroad4','crossroad5',
					'driving1','driving2','crossroad6','crossroad7','cropped_crossroad3','cropped_driving2']

	# dataset_list = ['cropped_crossroad3']
	short_video_length = 5*60 # divide each video into 5-min
	f = open('VideoStorm_motivation_final.csv','w')
	for dataset in dataset_list:
		height = image_resolution_dict[dataset][1]
		# load fasterRCNN + full resolution + highest frame rate as ground truth
		if 'highway' in dataset:
			frame_rate = 25
		else:
			frame_rate = 30							   
		standard_frame_rate = frame_rate
		gt_file = path + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv' 	
		gt, num_of_frames = load_full_model_detection(gt_file, height)	

		num_of_short_videos = num_of_frames//(short_video_length*frame_rate)

		for i in range(num_of_short_videos):
			start_frame = i * (short_video_length*frame_rate)
			end_frame = (i+1) * (short_video_length*frame_rate)


			# use 30 seconds video for profiling
			best_model, best_frame_rate = profile(dataset, 
												  frame_rate, 
												  gt, 
												  start_frame)
			# test on the whole video
			best_sample_rate = standard_frame_rate / best_frame_rate

			f1 = profile_eval(dataset, 
							  frame_rate,
							  gt,
							  best_model, 
							  best_sample_rate,
							  start_frame,
							  end_frame)

			print(dataset+str(i), best_frame_rate, f1)

			f.write(dataset + '_' + str(i) + ',' + str(best_model) + ',' + str(f1) + ',' + str(best_frame_rate/standard_frame_rate) + '\n')

			


if __name__ == '__main__':
	main()