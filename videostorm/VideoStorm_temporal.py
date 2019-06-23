
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
			img_index = int(line_list[0].split('.')[0]) - 1
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
					if t == 3 or t == 8:
						# if h > height/float(20):# object is car, this depends on the task
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
	# dataset_list = ['crossroad3','highway','crossroad2','driving2']
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

	dataset_list = frame_rate_dict.keys()
	fileID = open('VideoStorm_youtube_result_COCO.csv', 'w')
	# data_path = '/Users/zhujunxiao/Desktop/benchmarking/New_Dataset/'
	data_path = '/home/zhujun/video_analytics_pipelines/dataset/Youtube/'
	target_f1 = 0.9
	# different videos have different frame rate, to fairly compare across 
	# videos, we first change the frame rate to 10fps
	chunk_length = 30 # chunk a long video into 30-second short videos
	temporal_sampling_list = [20,15,10,5,4,3,2.5,2,1.8,1.5,1.2,1]

	# fig, ax = plt.subplots(1,1)
	# color=cm.rainbow(np.linspace(0,1,len(dataset_list)))
	# detail_f = open('detection_result.csv','w')


	for video_index in dataset_list:
		# run the full model on each frame first, and save in input_w_gt.csv 
		frame_rate = frame_rate_dict[video_index] 
		# standard_frame_rate = 10.0
		standard_frame_rate = 10
		height = image_resolution_dict[video_index][1]
		# fullmodel_detection_path = data_path + video_index + '/profile/updated_input_w_gt.csv'
		fullmodel_detection_path = data_path + video_index + '/profile/updated_gt_FasterRCNN_COCO.csv'
		full_model_dt, num_of_frames = load_full_model_detection(fullmodel_detection_path, height)
		F1_score_list = defaultdict(list)
		for sample_rate in temporal_sampling_list:
			img_cn = 0
			tp = defaultdict(int)
			fp = defaultdict(int)
			fn = defaultdict(int)
			save_dt = []

			for img_index in range(0, num_of_frames):
				dt_boxes_final = []
				current_full_model_dt = full_model_dt[img_index]
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


				# compute fn, tn, tp
				# for boxA in current_full_model_dt:
				# 	flag = 0
				# 	iou_list = []
				# 	for boxB in dt_boxes_final:
				# 		iou = IoU(boxA, boxB)
				# 		iou_list.append(iou)
				# 	if iou_list:
				# 		if max(iou_list) >= iou_thresh:
				# 			tp[img_index] += 1
				# 			dt_boxes_final.remove(dt_boxes_final[
				# 							iou_list.index(max(iou_list))])
				# 			flag = 1
				# 		#break
				# 	if not flag:
				# 		fn[img_index] += 1


				# fp[img_index] += len(dt_boxes_final)
				tp[img_index], fp[img_index], fn[img_index] = \
					eval_single_image(current_full_model_dt, dt_boxes_final)
					
			tp_total = defaultdict(int)
			fp_total = defaultdict(int)
			fn_total = defaultdict(int)
			for index in range(img_cn):
				key = index // int(chunk_length*standard_frame_rate)

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



		frame_rate_list = [standard_frame_rate/x 
						for x in temporal_sampling_list]

		for key in sorted(F1_score_list.keys()):
			current_f1_list = F1_score_list[key]
			# print(list(zip(frame_rate_list, current_f1_list)))

			if current_f1_list[-1] == 0:
				target_frame_rate = standard_frame_rate
			else:
				F1_score_norm = [x/current_f1_list[-1] for x in current_f1_list]
				index = next(x[0] for x in enumerate(F1_score_norm) 
									if x[1] > target_f1)
				if index == 0:
					target_frame_rate = frame_rate_list[0]
				else:
					point_a = (current_f1_list[index-1], frame_rate_list[index-1])
					point_b = (current_f1_list[index], frame_rate_list[index])


					target_frame_rate  = interpolation(point_a, point_b, target_f1)
			print(key, target_frame_rate)
			fileID.write(video_index+'_'+str(key)+','+str(target_frame_rate/standard_frame_rate)+'\n')

			# ax.plot(frame_rate_list, F1_score_norm,'-o',
			# 		c=color[dataset_list.index(video_index)])
	# 		detail_f.write(video_index+'_'+str(key)+','
	# 			+' '.join(str(x) for x in frame_rate_list)+','
	# 			+' '.join([str(x) for x in F1_score_list[key]])+'\n')
	# plt.show()


if __name__ == '__main__':
  main()