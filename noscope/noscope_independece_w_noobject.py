# test the independence between two modules of noscope
import cv2
import numpy as np
import os
import keras
import time
from collections import Counter, defaultdict
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import model_from_json, Model, load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import f1_score,confusion_matrix,classification_report,\
accuracy_score

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err



def difference_detector(image_path, img):
	'''
	Load a background image, then compute the difference (MSE) between current 
	image and background image. If difference < threshold, this frame should be
	skipped.
	'''
	# static_img = image_path + '/000056.jpg' # cropped_crossroad4
	static_img = image_path + '/000540.jpg' # cropped_crossroad4_2
	original = cv2.imread(img)
	original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
	static = cv2.imread(static_img)
	static = cv2.cvtColor(static, cv2.COLOR_BGR2GRAY)

	MSE = mse(original, static)

	# if MSE < thresh:
	# 	skip_flag = 1
	# else:
	# 	skip_flag = 0

	return MSE




def detect(model, img_path, test_labels, all_classes, frame_start, frame_end):
	def predict(model, img_filename):
		img =  load_img(img_filename,
						target_size= (227, 227))
		X = (img_to_array(img)).astype('float32') / 255
		X = np.expand_dims(X, axis=0)
		return model.predict(X)
	y_prob = []
	y_true = []
	for frame_id in range(frame_start, frame_end):
		img_name = img_path + '/' + format(frame_id, '06d') + '.jpg'
		tmp = predict(model, img_name)
		if frame_id in test_labels:
			y_prob.append(tmp[0])
			y_true.append(test_labels[frame_id])

	return y_prob, y_true


def para_sweep(y_prob, y_true, time_duration, frame_rate, all_classes):
	tmp_classes = all_classes + ['no_object']
	perf = []
	target_f1 = 0.9
	para_dict = {}
	for thresh_high in [0.7]:
		y_pred = np.empty(len(y_prob), dtype=int)
		y_pipeline = np.empty(len(y_prob), dtype=int)
		full_model_cn = 0
		for i in range(len(y_prob)):
			y_pred[i] = np.argmax(y_prob[i])
			if max(y_prob[i]) > thresh_high:
				y_pipeline[i] = y_pred[i]
			else:
				y_pipeline[i] = y_true[i]
				full_model_cn += 1
				# triggered_index.append(i)

		# 	print('predict, pipeline, true:', all_classes[y_pred[i]],  
		# 		tmp_classes[y_pipeline[i]], tmp_classes[y_true[i]])
		# print('Thresh:', thresh_high, thresh_low)
		# print('number of full model:', full_model_cn, 
		# 	full_model_cn/len(y_true))
		f1 = f1_score(y_true, y_pipeline,average='micro')
		perf.append((f1, full_model_cn))
		para_dict[(f1, full_model_cn)] = (thresh_high)

		# print('F1 score:', f1)
		# print('Small model F1:', f1_score(y_true, y_pred,average='micro'))
		# print(confusion_matrix(y_true, y_pred))
		# print(classification_report(y_true, y_pipeline))

	sorted_perf = sorted(perf, key = lambda x: x[0])
	diff = [abs(x-target_f1) for (x, _) in sorted_perf]
	best_f1 = sorted_perf[diff.index(min(diff))][0]
	best_full_model_cn = sorted_perf[diff.index(min(diff))][1]
	best_para = para_dict[(best_f1, best_full_model_cn)]
	print('best:', best_f1, best_full_model_cn/len(y_true), best_para)
	return (best_f1, best_full_model_cn)

def eval(model_path, label_file, image_path, class_info_path, 
	test_frame_range, test_frame_list):
	'''
	Based on the detection results and ground truth, compute the f1 score.
	'''
	test_labels, all_classes = load_data(class_info_path, 
										 test_frame_list,
										 label_file)
	print(model_path)
	model = load_model(model_path)
	frame_rate = 30
	time_duration = 30
	num_seg = len(test_frame_range) //(time_duration*frame_rate)
	perf = {}


	for seg_index in range(0, num_seg):
		frame_start = test_frame_range[0] + seg_index*(time_duration*frame_rate)
		frame_end = frame_rate * time_duration + frame_start
		print(frame_start, frame_end)
		y_prob, y_true = detect(model, 
							image_path,
							test_labels, 
							all_classes, 
							frame_start, 
							frame_end)
		(best_f1, best_full_model_cn) = para_sweep(y_prob, y_true, time_duration, 
							    		frame_rate, all_classes)
		print('seg_index:', seg_index, Counter(y_true))
		perf[seg_index] = (best_f1, best_full_model_cn/len(y_true))


	return perf


def load_data(class_info_path, test_frame_list, label_file):
	'''
	Given frame ids, load images for small model.
	'''
	def load_classes():
		all_classes = []
		with open(class_info_path, 'r') as f:
			for line in f:
				all_classes.append(line.strip())
		return all_classes
	all_classes = load_classes()
	print(all_classes)

	labels = {}
	test_labels = {}
	test_dist = defaultdict(int)
	with open(label_file, 'r') as f:
		for line in f:
			line_list = line.strip().split(',')
			labels[int(line_list[0])] = line_list[1].strip().replace('"','')\
										.replace(' ','_')
	for img_index in test_frame_list:
		# image_name = format(img_index, '06d') + '.jpg'
		label = labels[img_index]
		test_dist[label] += 1
		# if label == 'no_object':
		# 	continue
		# if label == 'no_object':
			# test_labels[img_index] = len(all_classes)
		# else:
		assert label in all_classes, print(label)

		test_labels[img_index] = all_classes.index(label)
	print(test_dist)
	return test_labels, all_classes




# config GPU
def setup(gpu):
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.8
	set_session(tf.Session(config=config))
	return



def main():
	gpu = '1'
	setup(gpu)
	dataset = 'cropped_crossroad4_2'
	image_path = '/home/zhujunxiao/video_analytics_pipelines/fast/data/' + dataset
	path = '/home/zhujunxiao/video_analytics_pipelines/fast/data/'
	label_file = os.path.join(path, 
							dataset+'_label_from_COCO_direct_filtered_new.csv')
	# model_path = '/home/zhujunxiao/tmp/' + dataset  + '.h5'
	# class_info_path = '/home/zhujunxiao/tmp/' + dataset   + '_allclasses.csv'

	model_path = './models/' + dataset  + '_alexnet_w_noobject2.h5'
	class_info_path = './info/' + dataset  +  '_alexnet_w_noobject_allclasses2.csv'
	test_frame_range = range(18001, 30001)


	test_labels, all_classes = load_data(class_info_path, 
										 test_frame_range,
										 label_file)	

	# check frame diff detector accuracy
	tp = 0
	fp = 0
	fn = 0
	test_images_after_difference_detector = []
	MSE_dict = {}
	for frame_id in test_frame_range:
		img_filename = image_path + '/' + format(frame_id, '06d') + '.jpg'
		MSE = difference_detector(image_path, img_filename)
		MSE_dict[frame_id] = MSE

	for thresh in [70]:
		for frame_id in test_frame_range:
			MSE = MSE_dict[frame_id]
			if MSE < thresh: 
				# should be skipped
				if test_labels[frame_id] == 'no_object':
					tp += 1
				else:
					fn += 1
				continue			
			else:
				# should be fed into small model
				test_images_after_difference_detector.append(frame_id)
				if test_labels[frame_id] != 'no_object':
					tp += 1
				else:
					fp += 1




		precision = tp/(tp+fp)
		recall = tp/(tp+fn)
		f1 = 2*(precision*recall)/(precision+recall)
		print(thresh, f1)




	# print(len(test_images_after_difference_detector), len(test_frame_range))
	# # # first test the small model performance without difference detector
	# perf = eval(model_path, label_file, image_path, 
	# 			  class_info_path, test_frame_range, test_images_after_difference_detector) 

	# perf2 = eval(model_path, label_file, image_path, 
	# 			  class_info_path, test_frame_range, test_frame_range)
	# with open('independence_test_' + dataset + '_0.7.csv', 'w') as f:
	# 	f.write('seg_index, after filter f1, after filter bw, before filter f1, before filter bw\n')

	# 	for key in sorted(perf.keys()):
	# 		print(key, perf[key], perf2[key])
	# 		f.write(str(key) + ',' + str(perf[key][0]) + ',' + str(perf[key][1]) + ',' +
	# 			str(perf2[key][0]) + ',' + str(perf2[key][1]) + '\n')


	return


if __name__ == '__main__':
	main()