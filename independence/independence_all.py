import pdb
import argparse
import csv
import sys
import numpy as np
import pdb
import argparse
import csv
import sys
import numpy as np
sys.path.append('/home/zhujunxiao/video_analytics_pipelines/code_repo/benchmarking/videostorm/')
from VideoStorm import VideoStorm
import copy
import os
sys.path.append('/home/zhujunxiao/video_analytics_pipelines/code_repo/benchmarking/')
from video import YoutubeVideo
from collections import defaultdict
# from utils.model_utils import eval_single_image
from utils.utils import interpolation, compute_f1, IoU
from collections import defaultdict
from utils.model_utils import eval_single_image
sys.path.append('/home/zhujunxiao/video_analytics_pipelines/code_repo/benchmarking/awstream/')
from Awstream import scale_boxes
from scipy.stats.stats import pearsonr   


temporal_sampling_list = [20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1]
resol_list = ['720p', '540p', '480p', '360p']
model_list = ['FasterRCNN', 'Resnet50', 'inception', 'mobilenet']
OFFSET = 0



def run_spatial_temporal(dataset, short_video_length):
	f_out = open('./results/Independence_' + dataset + '_spatial_wrt_temporal.csv', 'w')
	f_out.write("video_name, resolution, frame_rate, f1\n")

	original_resol = '720p'
	metadata_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/metadata.json'
	gt_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/' + original_resol + \
			  '/profile/updated_gt_FasterRCNN_COCO_no_filter.csv'
	print("processing", dataset, gt_file)
	original_video = YoutubeVideo(dataset, original_resol, metadata_file, gt_file,
						 None, True)
	frame_rate = original_video.frame_rate
	frame_count = original_video.frame_count
	chunk_frame_cnt = short_video_length * frame_rate
	num_of_chunks = (frame_count-OFFSET*frame_rate)//chunk_frame_cnt  
	gt = original_video.get_video_detection()

	dt = {}

	for resol in resol_list:
		dt_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/' + resol + \
				  '/profile/updated_gt_FasterRCNN_COCO_no_filter.csv'
		if not os.path.exists(dt_file):
			dt_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/' + resol + \
				  '/profile/updated_gt_FasterRCNN_COCO.csv'
		print(dataset, dt_file)
		video = YoutubeVideo(dataset, resol, metadata_file, dt_file,
							None, True)
		
		dt = video.get_video_detection()

		for i in range(num_of_chunks):
			clip = dataset + '_' + str(i)
			# the 1st frame in the chunk
			start_frame = i * chunk_frame_cnt + 1 + OFFSET * frame_rate
			# the last frame in the chunk
			end_frame = (i + 1) * chunk_frame_cnt + OFFSET * frame_rate


			video_name = dataset + '_' + str(i)
			print('short video {}, start={}, end={}'.format(video_name, 
															start_frame,
															end_frame))
			f1_list = []
			for sample_rate in temporal_sampling_list:
				tpos = defaultdict(int)     
				fpos = defaultdict(int)
				fneg = defaultdict(int)
				save_dt = []

				for img_index in range(start_frame, end_frame + 1):
					# based on sample rate, decide whether this frame is sampled
					if img_index % sample_rate >= 1:
						# this frame is not sampled, so ignore
						continue
					else:
						# this frame is sampled, so use the full model result
						current_gt = gt[img_index].copy()
						dt_boxes_final = dt[img_index].copy()
						current_gt = scale_boxes(current_gt, original_video.resolution,
								 video.resolution) 
						tpos[img_index], fpos[img_index], fneg[img_index] = \
							eval_single_image(current_gt, dt_boxes_final)

				tp_total = sum(tpos.values())
				fp_total = sum(fpos.values())
				fn_total = sum(fneg.values())

				f1_score = compute_f1(tp_total, fp_total, fn_total)
				f_out.write(','.join([video_name, resol, str(1/sample_rate),
										str(f1_score)])+'\n')
	
	return


def run_spatial_model(dataset, short_video_length):
	f_out = open('./results/Independence_' + dataset + '_spatial_wrt_model.csv', 'w')
	f_out.write("video_name, resolution, model, f1\n")
	metadata_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/metadata.json'
	for model in model_list:
		original_resol = '720p'
		gt_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/' + original_resol + \
					'/profile/updated_gt_' + model + '_COCO_no_filter.csv'
		original_video = YoutubeVideo(dataset, original_resol, metadata_file, gt_file,
									None, True)
		gt = original_video.get_video_detection() 
		frame_rate = original_video.frame_rate
		frame_count = original_video.frame_count
		chunk_frame_cnt = short_video_length * frame_rate
		num_of_chunks = (frame_count-OFFSET*frame_rate)//chunk_frame_cnt
		for resol in resol_list:
			dt_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/' + \
						resol + '/profile/updated_gt_' + model + '_COCO_no_filter.csv'
			video = YoutubeVideo(dataset, resol, metadata_file, dt_file,
								None, True)
			dt = video.get_video_detection() 
			for i in range(num_of_chunks):
				# the 1st frame in the chunk
				start_frame = i * chunk_frame_cnt + 1 + OFFSET * frame_rate
				# the last frame in the chunk
				end_frame = (i + 1) * chunk_frame_cnt + OFFSET * frame_rate
				video_name = dataset + '_' + str(i)
				print('short video {}, start={}, end={}'.format(video_name, 
																start_frame,
																end_frame))

				f1_list = []
				tpos = defaultdict(int)
				fpos = defaultdict(int)
				fneg = defaultdict(int)
				save_dt = []

				for img_index in range(start_frame, end_frame+1):
					# this frame is sampled, so use the full model result
					current_gt = copy.deepcopy(gt[img_index])
					dt_boxes_final = copy.deepcopy(dt[img_index])
					current_gt = scale_boxes(current_gt, original_video.resolution,
											video.resolution)        
					tpos[img_index], fpos[img_index], fneg[img_index] = \
								eval_single_image(current_gt, dt_boxes_final)

				tp_total = sum(tpos.values())
				fp_total = sum(fpos.values())
				fn_total = sum(fneg.values())

				f1_score = compute_f1(tp_total, fp_total, fn_total)
				print('resol={}, model={}, f1={}'.format(video.resolution, model, f1_score))
				f_out.write(','.join([video_name, resol, model,
										str(f1_score)])+'\n')
	
	return



def run_model_spatial(dataset, short_video_length):
	f_out = open('./results/Independence_' + dataset + '_model_wrt_spatial.csv', 'w')
	f_out.write("video_name, resolution, model,f1\n")
	metadata_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/metadata.json'
	for resol in resol_list:      
		for model in model_list:
			gt_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/' + \
					   resol + '/profile/updated_gt_FasterRCNN_COCO_no_filter.csv'
			original_video = YoutubeVideo(dataset, resol, metadata_file, gt_file,
										None, True)
			gt = original_video.get_video_detection() 
			frame_rate = original_video.frame_rate
			frame_count = original_video.frame_count
			chunk_frame_cnt = short_video_length * frame_rate
			num_of_chunks = (frame_count-OFFSET*frame_rate)//chunk_frame_cnt


			dt_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/' + \
						resol + '/profile/updated_gt_' + model + '_COCO_no_filter.csv'
			video = YoutubeVideo(dataset, resol, metadata_file, dt_file,
								None, True)
			dt = video.get_video_detection() 
			for i in range(num_of_chunks):
				clip = dataset + '_' + str(i)
				# the 1st frame in the chunk
				start_frame = i * chunk_frame_cnt + 1 + OFFSET * frame_rate
				# the last frame in the chunk
				end_frame = (i + 1) * chunk_frame_cnt + OFFSET * frame_rate
				video_name = dataset + '_' + str(i)
				print('short video {}, start={}, end={}'.format(video_name, 
																start_frame,
																end_frame))


				f1_list = []
				tpos = defaultdict(int)
				fpos = defaultdict(int)
				fneg = defaultdict(int)
				save_dt = []

				for img_index in range(start_frame, end_frame+1):
					# this frame is sampled, so use the full model result
					current_gt = copy.deepcopy(gt[img_index])
					dt_boxes_final = copy.deepcopy(dt[img_index])
					tpos[img_index], fpos[img_index], fneg[img_index] = \
								eval_single_image(current_gt, dt_boxes_final)


				tp_total = sum(tpos.values())
				fp_total = sum(fpos.values())
				fn_total = sum(fneg.values())

				f1_score = compute_f1(tp_total, fp_total, fn_total)
				print('resol={}, model={}, f1={}'.format(video.resolution, model, f1_score))
				f_out.write(','.join([video_name, resol, model,
										str(f1_score)])+'\n')
	return

def run_model_temporal(dataset, short_video_length):
	f_out = open('./results/Independence_' + dataset + '_model_wrt_temporal.csv', 'w')
	f_out.write("video_name, model, frame rate, f1\n")
	metadata_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/metadata.json'



	original_model = 'FasterRCNN'
	gt_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/720p/'  + \
			  '/profile/updated_gt_' + original_model + '_COCO_no_filter.csv'
	print("processing", dataset, gt_file)
	original_video = YoutubeVideo(dataset, '720p', metadata_file, gt_file,
						 None, True)
	frame_rate = original_video.frame_rate
	frame_count = original_video.frame_count
	chunk_frame_cnt = short_video_length * frame_rate
	num_of_chunks = (frame_count-OFFSET*frame_rate)//chunk_frame_cnt  
	gt = original_video.get_video_detection()

	dt = {}

	for model in model_list:
		dt_file = '/mnt/data/zhujun/dataset/Youtube/' + dataset + '/720p/' + \
				  '/profile/updated_gt_' + model + '_COCO_no_filter.csv'
		print(dataset, dt_file)
		video = YoutubeVideo(dataset, '720p', metadata_file, dt_file,
							None, True)
		dt = video.get_video_detection()
		for i in range(num_of_chunks):
			clip = dataset + '_' + str(i)
			# the 1st frame in the chunk
			start_frame = i * chunk_frame_cnt + 1 + OFFSET * frame_rate
			# the last frame in the chunk
			end_frame = (i + 1) * chunk_frame_cnt + OFFSET * frame_rate


			video_name = dataset + '_' + str(i)
			print('short video {}, start={}, end={}'.format(video_name, 
															start_frame,
															end_frame))
			f1_list = []
			for sample_rate in temporal_sampling_list:
				tpos = defaultdict(int)     
				fpos = defaultdict(int)
				fneg = defaultdict(int)
				save_dt = []

				for img_index in range(start_frame, end_frame + 1):
					# based on sample rate, decide whether this frame is sampled
					if img_index % sample_rate >= 1:
						# this frame is not sampled, so ignore
						continue
					else:
						# this frame is sampled, so use the full model result
						current_gt = gt[img_index].copy()
						dt_boxes_final = dt[img_index].copy()
						tpos[img_index], fpos[img_index], fneg[img_index] = \
							eval_single_image(current_gt, dt_boxes_final)

				tp_total = sum(tpos.values())
				fp_total = sum(fpos.values())
				fn_total = sum(fneg.values())

				f1_score = compute_f1(tp_total, fp_total, fn_total)
				f_out.write(','.join([video_name, model, str(1/sample_rate),
										str(f1_score)])+'\n')
	return






def load_temporal_spatial(dataset):
	cor_wrt_spatial = {}
	seg_names = []
	def _load_f1_gpu_curve(filename):
		perf = defaultdict(list)
		with open(filename, 'r') as f:
			f.readline()
			for line in f:
				line_list = line.strip().split(',')
				key = line_list[0]
				gpu = float(line_list[1])
				f1 = float(line_list[2])
				perf[key].append(f1)
		return perf
	temporal_spatial_f1 = defaultdict(list)
	for resol in resol_list:
		filename = './results/' + dataset + '_' + resol + '.csv'
		temporal_perf = _load_f1_gpu_curve(filename)
		for key in temporal_perf.keys():
			assert len(temporal_perf[key]) == len(temporal_sampling_list)
			temporal_spatial_f1[key].append(temporal_perf[key])
	

	for key in temporal_spatial_f1.keys():
		spatial_var = []
		perf = []
		for i in range(len(temporal_sampling_list)):
			for j in range(len(resol_list)):
				spatial_var.append(j) # resol index
				perf.append(temporal_spatial_f1[key][j][i])
		# print(key, pearsonr(spatial_var, perf))
		seg_names.append(key)
		cor_wrt_spatial[key] = pearsonr(spatial_var, perf)[0]
	return seg_names, cor_wrt_spatial
			

def load_temporal_model(dataset):
	cor_wrt_model = {}
	seg_names = []
	def _load_f1_gpu_curve(filename):
		perf = defaultdict(list)
		with open(filename, 'r') as f:
			f.readline()
			for line in f:
				line_list = line.strip().split(',')
				key = line_list[0]
				gpu = float(line_list[1])
				f1 = float(line_list[2])
				perf[key].append(f1)
		return perf
	temporal_model_f1 = defaultdict(list)
	for model in model_list:
		filename = './results/' + dataset + '_720p_' + model + '.csv'
		temporal_perf = _load_f1_gpu_curve(filename)
		for key in temporal_perf.keys():
			assert len(temporal_perf[key]) == len(temporal_sampling_list)
			temporal_model_f1[key].append(temporal_perf[key])
	

	for key in temporal_model_f1.keys():
		model_var = []
		perf = []
		for i in range(len(temporal_sampling_list)):
			for j in range(len(model_list)):
				model_var.append(j) # resol index
				perf.append(temporal_model_f1[key][j][i])
		seg_names.append(key)
		cor_wrt_model[key] = pearsonr(model_var, perf)[0]
	return seg_names, cor_wrt_model
	


def load_spatial_temporal(dataset):
	filename = './results/Independence_' + dataset + '_spatial_wrt_temporal.csv'
	spatial_temporal_f1 = defaultdict(list)
	cor_spatial_wrt_temporal = {}
	seg_names = []
	with open(filename, 'r') as f:
		f.readline()
		for line in f:
			line_list = line.strip().split(',')
			key = line_list[0]
			resol = line_list[1]
			frame_rate = float(line_list[2])
			f1 = float(line_list[3])
			spatial_temporal_f1[key].append((resol, frame_rate, f1))
	
	for key in spatial_temporal_f1.keys():
		temporal_var = []
		perf = []
		for (resol, frame_rate, f1) in spatial_temporal_f1[key]:
			temporal_var.append(frame_rate)
			perf.append(f1)

		seg_names.append(key)
		cor_spatial_wrt_temporal[key] = pearsonr(temporal_var, perf)[0]
	return seg_names, cor_spatial_wrt_temporal


def load_spatial_model(dataset):
	filename = './results/Independence_' + dataset + '_spatial_wrt_model.csv'
	spatial_model_f1 = defaultdict(list)
	cor_spatial_wrt_model = {}
	seg_names = []
	with open(filename, 'r') as f:
		f.readline()
		for line in f:
			line_list = line.strip().split(',')
			key = line_list[0]
			resol = line_list[1]
			model = model_list.index(line_list[2])
			f1 = float(line_list[3])
			spatial_model_f1[key].append((resol, model, f1))
	
	for key in spatial_model_f1.keys():
		model_var = []
		perf = []
		for (resol, model, f1) in spatial_model_f1[key]:
			model_var.append(model)
			perf.append(f1)

		seg_names.append(key)
		cor_spatial_wrt_model[key] = pearsonr(model_var, perf)[0]
		print(key, pearsonr(model_var, perf))
	return seg_names, cor_spatial_wrt_model 

def load_model_temporal(dataset):
	filename = './results/Independence_' + dataset + '_model_wrt_temporal.csv'
	model_temporal_f1 = defaultdict(list)
	cor_model_wrt_temporal = {}
	seg_names = []
	with open(filename, 'r') as f:
		f.readline()
		for line in f:
			line_list = line.strip().split(',')
			key = line_list[0]
			model = line_list[1]
			frame_rate = float(line_list[2])
			f1 = float(line_list[3])
			model_temporal_f1[key].append((model, frame_rate, f1))
	
	for key in model_temporal_f1.keys():
		temporal_var = []
		perf = []
		for (model, frame_rate, f1) in model_temporal_f1[key]:
			temporal_var.append(frame_rate)
			perf.append(f1)

		seg_names.append(key)
		cor_model_wrt_temporal[key] = pearsonr(temporal_var, perf)[0]	
	return seg_names, cor_model_wrt_temporal

def load_model_spatial(dataset):
	filename = './results/Independence_' + dataset + '_model_wrt_spatial.csv'
	model_spatial_f1 = defaultdict(list)
	cor_model_wrt_spatial = {}
	seg_names = []
	with open(filename, 'r') as f:
		f.readline()
		for line in f:
			line_list = line.strip().split(',')
			key = line_list[0]
			model = line_list[2]
			resol = resol_list.index(line_list[1])
			f1 = float(line_list[3])
			model_spatial_f1[key].append((model, resol, f1))
	
	for key in model_spatial_f1.keys():
		spatial_var = []
		perf = []
		for (model, resol, f1) in model_spatial_f1[key]:
			spatial_var.append(resol)
			perf.append(f1)

		seg_names.append(key)
		cor_model_wrt_spatial[key] = pearsonr(spatial_var, perf)[0]	
		print(key, pearsonr(spatial_var, perf))
	return seg_names, cor_model_wrt_spatial




def main():
	# set up
	short_video_length = 30
	f = open('independence_correlation.txt', 'w')
	f.write('# correlation, median, std\n')
	temporal_wrt_spatial = []
	temporal_wrt_model = []
	spatial_wrt_temporal = []
	spatial_wrt_model = []
	model_wrt_temporal = []
	model_wrt_spatial = []
	# for dataset in ['drift']:  #['motorway', 'crossroad3', 'crossroad4', 'drift']

	# 	# run independence test
	# 	# run_temporal_spatial() =  run_temporal_wrt_spatial.sh
	# 	# run_temporal_model() = run_temporal_wrt_model.sh


		# run_spatial_temporal(dataset, short_video_length)
		# run_spatial_model(dataset, short_video_length)
		# run_model_spatial(dataset, short_video_length)
		# run_model_temporal(dataset, short_video_length)



	for dataset in ['crossroad3', 'motorway', 'drift']:



		_, cor_temporal_wrt_spatial = load_temporal_spatial(dataset)
		temporal_wrt_spatial += cor_temporal_wrt_spatial.values()
		# print(cor_temporal_wrt_spatial)
		seg_names, cor_temporal_wrt_model = load_temporal_model(dataset)
		temporal_wrt_model += cor_temporal_wrt_model.values()
		# print(cor_temporal_wrt_model)
		seg_names, cor_spatial_wrt_temporal = load_spatial_temporal(dataset)
		spatial_wrt_temporal += cor_spatial_wrt_temporal.values()
		# print(cor_spatial_wrt_temporal)
		seg_names, cor_spatial_wrt_model = load_spatial_model(dataset)
		spatial_wrt_model += cor_spatial_wrt_model.values()
		# print(cor_spatial_wrt_model)
		seg_names, cor_model_wrt_temporal = load_model_temporal(dataset)
		model_wrt_temporal += cor_model_wrt_temporal.values()
		# print(cor_model_wrt_temporal)
		seg_names, cor_model_wrt_spatial = load_model_spatial(dataset)
		model_wrt_spatial += cor_model_wrt_spatial.values()
		# print(cor_model_wrt_spatial)
		
	f.write(','.join(['T_wrt_S', str(np.mean(temporal_wrt_spatial)), str(np.std(temporal_wrt_spatial))]) + '\n')
	f.write(','.join(['T_wrt_M', str(np.mean(temporal_wrt_model)), str(np.std(temporal_wrt_model))]) + '\n')	
	f.write(','.join(['S_wrt_T', str(np.mean(spatial_wrt_temporal)), str(np.std(spatial_wrt_temporal))]) + '\n')	
	f.write(','.join(['S_wrt_M', str(np.mean(spatial_wrt_model)), str(np.std(spatial_wrt_model))]) + '\n')	
	f.write(','.join(['M_wrt_T', str(np.mean(model_wrt_temporal)), str(np.std(model_wrt_temporal))]) + '\n')	
	f.write(','.join(['M_wrt_S', str(np.mean(model_wrt_spatial)), str(np.std(model_wrt_spatial))]) + '\n')	



	return

if __name__ == '__main__':
	main()