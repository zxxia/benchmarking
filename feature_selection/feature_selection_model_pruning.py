from matplotlib import pyplot as plt
import glob
from collections import defaultdict
import os
from scipy import spatial



train_partition = [1, 18001]
all_classes = ['car','person','truck','bicycle','bus','motorcycle','no_object']

def easy_frame(annot, large_object_thresh=0.2):
	if annot[0] == 'no_object':
		return True
	elif float(annot[1]) >= large_object_thresh:
		return True
	else:
		return False

def load_ground_truth(ground_truth_file):
	gt = {}
	with open(ground_truth_file, 'r') as f:
		for line in f:
			line_list = line.strip().split(',')
			frame_index = int(line_list[0])
			label =line_list[1]
			assert label in all_classes, print(label, line)
			gt[frame_index] = [label]
			if len(line_list) > 2:
				area = float(line_list[2])
				confidence = float(line_list[3])
				gt[frame_index] = [label, area, confidence]
	return gt

def compute_para(gt, dataset, train_data_dist, short_video_length=30, frame_rate=30):
	# compute two paras
	# 1. percentage of easy frames; 
	# 2. similarity between traininga and test data
	train_data_label = defaultdict(int)
	for i in range(train_partition[0], train_partition[1]):
		train_data_label[gt[i][0]] += 1

	if train_data_dist == []:
		for object_class in all_classes:
			train_data_dist.append(train_data_label[object_class])	

	print(train_data_dist)
	easy_frame_dict = defaultdict(list)
	data_label = defaultdict(list)
	similarity = {}
	percentage_w_easy_frame = {}
	for key in sorted(gt.keys()):
		seg_index = key//(short_video_length*frame_rate)
		easy_frame_dict[dataset+'_'+str(seg_index)].append(easy_frame(gt[key]))
		data_label[dataset+'_'+str(seg_index)].append(gt[key][0])

	for key in data_label.keys():
		current_data_label = data_label[key]
		current_easy_frame = easy_frame_dict[key]
		current_dist = []
		for object_class in all_classes:
			current_dist.append(current_data_label.count(object_class))
		similarity[key] =  1 - spatial.distance.cosine(train_data_dist, 
												  current_dist)
		percentage_w_easy_frame[key] = current_easy_frame.count(True)/len(current_easy_frame)
		print(key, similarity[key], current_dist)

	return percentage_w_easy_frame, similarity



def load_model_selection_perf(perf_file, dataset, gt, short_video_length=30, frame_rate=30):
	tp = defaultdict(int)
	total_cn = defaultdict(int)
	acc = {}
	acc_per_seg = defaultdict(list)
	with open(perf_file, 'r') as f:
		for line in f:
			line_list = line.strip().split(',')
			img_index = int(line_list[0])
			seg_index = img_index//(short_video_length*frame_rate)
			label = line_list[2]
			key = ('mobilenet', dataset + '_' + str(seg_index))
			if label == gt[img_index][0]:
				tp[key] += 1
			total_cn[key] += 1


			label = line_list[3]
			key = ('inception', dataset + '_' + str(seg_index))
			if label == gt[img_index][0]:
				tp[key] += 1
			total_cn[key] += 1


			label = line_list[4]
			key = ('resnet50', dataset + '_' + str(seg_index))
			if label == gt[img_index][0]:
				tp[key] += 1
			total_cn[key] += 1

	seg_name = []
	for key in sorted(tp.keys()):
		acc[key] = tp[key]/total_cn[key]
		if key[1] not in seg_name:
			seg_name.append(key[1])

	for seg in seg_name:
		for model in ['mobilenet', 'inception', 'resnet50']:
			acc_per_seg[seg].append(acc[(model, seg)])
		acc_per_seg[seg].append(1)


	return acc, acc_per_seg

def load_noscope_perf(perf_file, gt, dataset, short_video_length=30, frame_rate=30):
	tp_cn = defaultdict(int)
	full_model_cn = defaultdict(int)
	total_cn = defaultdict(int)
	acc_pipeline = {}
	gpu = {}
	small_model_result = {}
	small_model_speed = 2.6
	full_model_speed = 100
	with open(perf_file, 'r') as f:
		f.readline()
		for line in f:
			line_list = line.strip().split(',')
			frame_index = int(line_list[0])
			y_small_model = line_list[2]
			confidence_score = float(line_list[3]) 
			small_model_result[frame_index] = (y_small_model, confidence_score)

	thresh = 0.8
	for frame_index in sorted(small_model_result.keys()):
		seg_index = frame_index//(short_video_length*frame_rate)
		confidence_score = small_model_result[frame_index][1]
		y_small_model = small_model_result[frame_index][0]
		if confidence_score < thresh:
			y_pipeline = gt[frame_index][0]
			full_model_cn[seg_index] += 1
		else:
			y_pipeline = y_small_model
		total_cn[seg_index] += 1
		if y_pipeline == gt[frame_index][0]:
			tp_cn[seg_index] += 1

	for key in tp_cn.keys():
		seg_name = dataset + '_' + str(key)
		acc_pipeline[seg_name] = tp_cn[key]/total_cn[key]
		gpu[seg_name] = (small_model_speed*(total_cn[key]-full_model_cn[key])/full_model_speed + full_model_cn[key])/total_cn[key]

	return acc_pipeline, gpu


def show_model_selection_fix_para_result(model_selection_acc, percentage_w_easy_frame, para):
	speed = {'mobilenet': 0.48, 'inception': 0.67, 
			 'resnet50': 0.88, 'FasterRCNN': 1}
	fig, ax = plt.subplots(1,2, sharex=True)
	for key in model_selection_acc.keys():
		if key[0] != para:
			continue
		else:
			ax[0].scatter(percentage_w_easy_frame[key[1]], 
						  model_selection_acc[key], c='b')
			ax[1].scatter(percentage_w_easy_frame[key[1]], 
						  speed[para], c='b')
	ax[0].set_ylabel('F1 score') 
	ax[1].set_xlabel('% of easy frames')
	ax[1].set_ylabel('GPU processing time') 
	plt.title(para)
	plt.show()
	return


def show_target_f1_result(acc_per_seg, percentage_w_easy_frame, target_f1):
	fig, ax = plt.subplots(1,1)
	gpu_list = [0.48, 0.67, 0.88, 1]
	for key in acc_per_seg:
		f1_list = acc_per_seg[key]
		index = next(x[0] for x in enumerate(f1_list) if x[1] >= target_f1)
		gpu = gpu_list[index]
		ax.scatter(percentage_w_easy_frame[key], gpu, c='b')
		# ax.text(percentage_w_easy_frame[key], gpu, key)
	ax.set_xlabel('% of easy frames')
	ax.set_ylabel('GPU processing time') 
	plt.show()

	return

def main():
	similarity = []
	cost = []
	target_f1 = 0.85
	path = '/Users/zhujunxiao/Desktop/benchmarking/Final_code/fast/'
	filename = '_car_truck_separate'

	# dataset = 'cropped_crossroad4'
	train_dist_dict = {
	'cropped_crossroad4_2': [2860, 621, 2824, 78, 0, 2, 11615]
#[0, 621, 5686, 78, 0, 2, 11613]
	}
	model_selection_acc_all = {}
	acc_per_seg_all = {}
	percentage_w_easy_frame_all = {}
	similarity_all = {}

	new_similarity = {}
	with open('cropped_crossroad4_similarity_simple_greyscale_normalized.csv', 'r') as f:
		for line in f:
			line_list = line.strip().split(',')
			key = line_list[0]
			new_similarity[key] = float(line_list[-1])
	with open('cropped_crossroad4_2_similarity.csv', 'r') as f:
		for line in f:
			line_list = line.strip().split(',')
			key = line_list[0]
			new_similarity[key] = float(line_list[-1])



	fig, ax = plt.subplots(1,2)
	for dataset in ['cropped_crossroad4', 'cropped_crossroad4_2','cropped_crossroad5',
					'cropped_driving2']:
		if dataset in train_dist_dict:
			train_data_dist = train_dist_dict[dataset]
		else:
			train_data_dist = []
		gt_file = path + 'label/' + dataset + '_ground_truth' + filename + '.csv'
		gt = load_ground_truth(gt_file)
		# compute content-level parameters
		percentage_w_easy_frame, similarity = compute_para(gt, dataset, 
														   train_data_dist)

		# load model selection performance
		model_selection_file = path + 'label/' + dataset + '_model_predictions.csv'
		model_selection_acc, acc_per_seg = load_model_selection_perf(model_selection_file, 
																	 dataset, 
																	 gt)

		# load noscope performance
	# 	noscope_file = path + 'noscope_small_model_predicted_' + dataset + filename + '.csv'
	# 	acc_pipeline, gpu = load_noscope_perf(noscope_file, gt, dataset)
	# 	for key in sorted(acc_pipeline.keys()):
	# 		if key not in new_similarity:
	# 			continue
	# 		print(key, new_similarity[key], acc_pipeline[key])

	# 		ax[0].scatter(new_similarity[key], acc_pipeline[key], c='b')
	# 		ax[0].text(new_similarity[key], acc_pipeline[key], key)

	# 		ax[1].scatter(new_similarity[key], gpu[key], c='b')
	# 		ax[1].text(new_similarity[key], gpu[key], key)


	# plt.show()
		# update the dictionaries
		model_selection_acc_all = {**model_selection_acc_all, **model_selection_acc}
		acc_per_seg_all = {**acc_per_seg_all, **acc_per_seg}
		percentage_w_easy_frame_all = {**percentage_w_easy_frame_all, 
									   **percentage_w_easy_frame}
		similarity_all = {**similarity_all, **similarity}

	# show model selection results
	for para in ['mobilenet', 'inception', 'resnet50']:
		show_model_selection_fix_para_result(model_selection_acc_all, 
											 percentage_w_easy_frame_all,
											 para)

	show_target_f1_result(acc_per_seg_all, percentage_w_easy_frame_all, target_f1)
	
	# show noscope results





	return


if __name__ == '__main__':
	main()