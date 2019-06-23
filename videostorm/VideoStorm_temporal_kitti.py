import time
import os
from collections import defaultdict
from my_utils import IoU, interpolation



def load_full_model_detection(fullmodel_detection_path):
	full_model_dt = {}
	gt = {}
	img_list = []
	with open(fullmodel_detection_path, 'r') as f:
		for line in f:
			line_list = line.strip().split(',')
			img_index = int(line_list[0].split('.')[0])

			# if img_index > 5000: # test on ~3 mins
			#   break
			if not line_list[1]: # no detected object
				dt_boxes_final = []
			else:
				dt_boxes_final = []
				dt_boxes = line_list[1].split(';')
				for dt_box in dt_boxes:
					# t is object type
					[x, y, w, h, t] = [int(i) for i in dt_box.split(' ')]
					if t == 1: # object is car, this depends on the task
						dt_boxes_final.append([x, y, x+w, y+h])

			# load the ground truth
			if not line_list[2]:
				gt_boxes_final = []
			else:
				gt_boxes_final = []
				gt_boxes = line_list[2].split(';')
				for gt_box in gt_boxes:
					# t is object type
					[x, y, w, h, t] = [int(i) for i in gt_box.split(' ')]
					if t == 1: # object is car, this depends on the task
						gt_boxes_final.append([x, y, x+w, y+h])			
			
			img_list.append(img_index)
			full_model_dt[img_index] = dt_boxes_final
			gt[img_index] = gt_boxes_final
			
	return full_model_dt, gt, img_list




def main():
	iou_thresh = 0.5
	path = '/home/zhujun/video_analytics_pipelines/dataset/KITTI/'
	video_index_dict = {'City':[1,2,5,9,11,13,14,17,18,48,51,56,57,59,60,84,91,93],
						'Road':[15,27,28,29,32,52,70],
						'Residential':[19,20,22,23,35,36,39,46,61,64,79,86,87]}
	temporal_sampling_list = [20,15,10,5,4,3,2.5,2,1.8,1.5,1.2,1]
	target_f1 = 0.9
	# different videos have different frame rate, to fairly compare across 
	# videos, we first change the frame rate to 10fps
	# KITTI already has frame rate 10fps
	standard_frame_rate = 10.0



	fileID = open('VideoStorm_Performance_KITTI.csv', 'w')
	f = open(path + 'VideoStorm_Performance_curve_KITTI.txt','w')

	for video_name in video_index_dict.keys():

		for video_index in video_index_dict[video_name]:
			fullmodel_detection_path = path + video_name + \
			'/2011_09_26_drive_' + format(video_index, '04d') + \
			'_sync/result/input_w_gt.csv'
			print(video_name, video_index)
			full_model_dt, gt, img_list = \
							load_full_model_detection(fullmodel_detection_path)
			F1_list = []

			frame_rate_list = [standard_frame_rate/x 
							for x in temporal_sampling_list]

			f.write(str(video_index) + ',' + ' '.join(str(x) for x in frame_rate_list) + ',')



			for sample_rate in temporal_sampling_list:
			# for sample_rate in [20,10,5,2,1]:
				tp = 0
				fp = 0
				fn = 0
				saved_dt =[]
				#tf.logging.set_verbosity(tf.logging.INFO)
				#saved_boxes = 
				#tfrecord_path = '/home/zhujun/video_analytics_pipelines/dataset/Caltech/set00/V014/detection.record'
				#tfrecord_path =  '/home/zhujun/video_analytics_pipelines/dataset/DukeMTMC/Output/detection_small_w_negative.record'
				for img_index in img_list:
					current_full_model_dt = full_model_dt[img_index]
					current_gt = gt[img_index]

					if img_index%sample_rate >= 1:
						dt_boxes_final = [box for box in saved_dt]

					else:
						dt_boxes_final = [box for box in current_full_model_dt]
						saved_dt = [box for box in dt_boxes_final]




					# compute true positive, false negative and false positive 
					for boxA in current_gt:
						flag = 0
						iou_list = []
						for boxB in dt_boxes_final:
							iou = IoU(boxA, boxB)
							iou_list.append(iou)
						if iou_list:	
							if max(iou_list) >= iou_thresh:
								tp += 1
								dt_boxes_final.remove(dt_boxes_final[
											iou_list.index(max(iou_list))])								
								flag = 1
						if not flag:
							fn += 1


					# for boxA in current_gt:
					# 	flag = 0
					# 	for boxB in dt_boxes_final:
					# 		iou = IoU(boxA, boxB)
					# 		if iou >= iou_thresh:	
					# 			tp += 1
					# 			dt_boxes_final.remove(boxB)								
					# 			flag = 1
					# 			break
					# 	if not flag:
					# 		fn += 1

					fp += len(dt_boxes_final)


				print("True positive:{0}, False positive: {1}, " 
					 "False negative: {2}".format(tp, fp, fn))


				if tp == 0:
					f1 = 0
				else:
					precison = float(tp) / (tp + fp)
					recall = float(tp) / (tp + fn)
					f1 = 2*(precison*recall)/(precison+recall) 
					print("sample rate: {3}, precison:{0}, Recall:{1},"
					" F1 score: {2}".format(precison, recall, f1, sample_rate))          
				F1_list.append(f1)
			print(F1_list)
			# f.write(tfrecord_path + '\n')
			f.write(' '.join([str(x) for x in F1_list])+'\n')

			value = F1_list
			if value[-1] == 0:
				target_frame_rate = standard_frame_rate
			else:
				result_vec = [x/value[-1] for x in value]
				index = next(x[0] for x in enumerate(result_vec) 
								  if x[1] > target_f1)
				if index == 0:
					target_frame_rate = frame_rate_list[0]
				else:
					point_a = (result_vec[index-1], frame_rate_list[index-1])
					point_b = (result_vec[index], frame_rate_list[index])


					target_frame_rate  = interpolation(point_a, point_b, target_f1)

			fileID.write(format(video_index, '04d')+','+str(target_frame_rate)+'\n')



if __name__ == '__main__':
	main()