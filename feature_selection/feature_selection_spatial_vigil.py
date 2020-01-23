import os
from collections import defaultdict
import glob
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn import preprocessing, metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import (LinearRegression, Ridge, 
								  Lasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
# from minepy import MINE
import operator
from mpl_toolkits.mplot3d import Axes3D



all_feature_names = [
'object_cn_median','object_cn_avg','object_cn_mode','object_cn_var','object_cn_std',
'object_cn_skewness', 'object_cn_kurtosis','object_cn_second_moment', 'object_cn_percentile5',
'object_cn_percentile25','object_cn_percentile75','object_cn_percentile95', 'object_cn_iqr',
'object_cn_entropy',
'object_size_median', 'object_size_avg', 'object_size_mode', 
'object_size_var', 'object_size_std', 'object_size_skewness', 'object_size_kurtosis', 
'object_size_second_moment', 'object_size_percentile5', 'object_size_percentile25', 
'object_size_percentile75', 'object_size_percentile95', 'object_size_iqr', 
'object_size_entropy', 'arrival_rate_median', 'arrival_rate_avg', 
'arrival_rate_mode', 'arrival_rate_var', 'arrival_rate_std', 'arrival_rate_skewness',
 'arrival_rate_kurtosis', 'arrival_rate_second_moment', 
 'arrival_rate_percentile5', 'arrival_rate_percentile25', 
 'arrival_rate_percentile75', 'arrival_rate_percentile95', 'arrival_rate_iqr', 
 'arrival_rate_entropy', 'velocity_median', 'velocity_avg', 'velocity_mode', 
 'velocity_var', 'velocity_std', 'velocity_skewness', 'velocity_kurtosis', 
 'velocity_second_moment', 'velocity_percentile5', 'velocity_percentile25', 
 'velocity_percentile75', 'velocity_percentile95', 'velocity_iqr', 
 'velocity_entropy', 'total_area_median', 'total_area_avg', 'total_area_mode', 
 'total_area_var', 'total_area_std', 'total_area_skewness', 'total_area_kurtosis', 
 'total_area_second_moment', 'total_area_percentile5', 'total_area_percentile25', 
 'total_area_percentile75', 'total_area_percentile95', 'total_area_iqr', 
 'total_area_entropy', 'percentage', 'percentage_w_new_objects']


video_to_delete = ['nyc', 'russia', 'crossroad', 'crossroad2', 'driving_downtown',
'tw','tw_road','tw_under_bridge','tw1','lane_split','crossroad3']
# video_to_delete = ['split', 'russia', 'bridge']
video_to_delete =[]
video_to_delete = ['nyc', 'russia',  'crossroad2', 'driving_downtown',
'tw_road','tw_under_bridge','tw1','tw','crossroad3','driving1']
def read_feature(feature_file):
	features = {}
	with open(feature_file) as f:
		f.readline()
		for line in f:
			line_list = line.strip().split(',')
			key = line_list[0]
			features[key] = [float(x) for x in line_list[1:]]
	return features


def Parser(pipeline, path):

	def AW_parser(path):
		perf = defaultdict(list)
		for file in glob.glob(path+'awstream_*.csv'):
			with open(file, 'r') as f:
				f.readline()
				for line in f:
					line_list = line.strip().split(',')
					dataset_name = line_list[0].replace('_' + \
										line_list[0].split('_')[-1], '')
					if dataset_name in video_to_delete:
						continue
					f1 = float(line_list[2])
					gpu = float(line_list[4])
					perf[line_list[0]]= (gpu, f1)	
		return perf

	def Vigil_parser(path):
		perf = defaultdict(list)
		for file in glob.glob(path+'vigil_*.csv'):
			with open(file, 'r') as f:
				f.readline()
				for line in f:
					line_list = line.strip().split(',')
					dataset_name = line_list[0].replace('_' + \
									line_list[0].split('_')[-1], '')
					if dataset_name in video_to_delete:
						continue
					f1 = float(line_list[1])
					gpu = float(line_list[2])
					area = float(line_list[3])
					true_area = float(line_list[4])
					perf[line_list[0]]= (gpu, f1, area, true_area)
		return perf 

	if pipeline == 'videostorm':
		perf = VS_parser(path)
	elif pipeline == 'glimpse':
		perf = GL_parser(path)
	elif pipeline == 'awstream':
		perf = AW_parser(path)
	elif pipeline == 'vigil':
		perf = Vigil_parser(path)
	else:
		print('Pipeline {} does not exist!!'.format(pipeline))

	return perf


def profile_parser(path):
	total_object_cn = {}
	for file in glob.glob(path+'awstream_*.csv'):
		with open(file, 'r') as f:
			f.readline()
			for line in f:
				line_list = line.strip().split(',')
				if line_list[1] != '720p':
					continue
				else:
					total_object_cn[line_list[0]] = int(line_list[4])
	return total_object_cn

def topK_index(data, K):
	indices = data.argsort()[-1*K:][::-1]
	return indices, data[indices]


def main():
	feature1 = 'percentage'
	feature2 = 'total_area_avg'
	feature3 = 'percentage'
	index1 = all_feature_names.index(feature1)
	index2 = all_feature_names.index(feature2)
	index3 = all_feature_names.index(feature3)

	feature_file = '/Users/zhujunxiao/Desktop/benchmarking/vldb/data/features_all_type_width_height_filter.csv'
	features = read_feature(feature_file)

	path = '/Users/zhujunxiao/Desktop/benchmarking/vldb/data/awstream/spatial_overfitting_results_09_28/'
	pipeline = 'awstream'
	awstream_perf = Parser(pipeline, path)

	path = '/Users/zhujunxiao/Desktop/benchmarking/vldb/data/awstream/spatial_overfitting_profile_09_28/'
	total_object_cn = profile_parser(path)

	awstream_perf = {}
	with open('./awstream/awstream_selected_video_resol_0.86.csv', 'r') as f:
		f.readline()
		for line in f:
			line_list = line.strip().split(',')
			dataset_name = line_list[0].replace('_' + line_list[0].split('_')[-1], '')
			if dataset_name in video_to_delete:
				# print(dataset_name)
				continue
			key = line_list[0]
			resol = int(line_list[2].replace('p', ''))
			f1 = float(line_list[3])
			bw = float(line_list[4])
			awstream_perf[key] = (bw, f1, resol)



	# path = '/Users/zhujunxiao/Desktop/benchmarking/vldb/data/vigil/mobilenet_remove_small_boxes/'

	# local detector: mobilenet  then faster rcnn on server
	path = '/Users/zhujunxiao/Desktop/benchmarking/vldb/data/vigil/vigil_mobilenet_results_10_11/'
	pipeline = 'vigil'
	vigil_perf = Parser(pipeline, path)

	y = []
	y1 = []

	X = []
	label = {}
	cn = 0
	fig = plt.figure()
	ax0 = fig.add_subplot(131)
	ax1 = fig.add_subplot(132)
	ax2 = fig.add_subplot(133)
	# ax2.set_xscale('log')
	# ax2.set_yscale('log')
	for key in sorted(vigil_perf.keys()):
		if key not in features:
			continue
		# if the number of objects is too small, the results will be easily biased
		# by one or two objects
		# if features[key][0] <= 0 or features[key][all_feature_names.index('percentage')] < 0.1:
		# 	continue


		if key not in awstream_perf:
			print('no awstream result for {}'.format(key))
			continue
		if features[key][all_feature_names.index('velocity_avg')] < 1 or \
			features[key][all_feature_names.index('object_size_avg')] <= 0:
			continue

		if  total_object_cn[key] < 200:
			continue
		# if features[key][all_feature_names.index('percentage')] < 0.3:
		# 	continue
		thresh1 = 0.1

		if np.abs(vigil_perf[key][1] - 0.9) > thresh1:
			continue 


		X.append(features[key])
		label[cn] = key
		cn += 1
		# print(key, vigil_perf[key])
		y.append(vigil_perf[key][0])
		y1.append(awstream_perf[key][0])
		ax0.scatter(features[key][all_feature_names.index(feature1)], awstream_perf[key][0], c='r')
		
		ax0.scatter(features[key][all_feature_names.index(feature1)], vigil_perf[key][0], c='b')

		ax1.scatter(features[key][all_feature_names.index(feature2)], awstream_perf[key][0], c='r')
		ax1.scatter(features[key][all_feature_names.index(feature2)], vigil_perf[key][0], c='b')
		# ax1.scatter(features[key][all_feature_names.index(feature2)], vigil_perf[key][2], c='b')
		# ax1.scatter(features[key][all_feature_names.index(feature2)], vigil_perf[key][3], c='k')
		# ax1.text(features[key][all_feature_names.index(feature2)], vigil_perf[key][0], key)
		# ax1.text(features[key][all_feature_names.index(feature2)], vigil_perf[key][2], key)
		# ax1.text(features[key][all_feature_names.index(feature2)], vigil_perf[key][3], key)
		thresh2 = 0.0
		if awstream_perf[key][0] - vigil_perf[key][0] > thresh2:
			ax2.scatter(features[key][all_feature_names.index(feature1)], features[key][all_feature_names.index(feature2)], c='b')
		elif vigil_perf[key][0] - awstream_perf[key][0] > thresh2:
			ax2.scatter(features[key][all_feature_names.index(feature1)], features[key][all_feature_names.index(feature2)], c='r')
		else:
			ax2.scatter(features[key][all_feature_names.index(feature1)], features[key][all_feature_names.index(feature2)], c='k')
		# ax2.text(features[key][all_feature_names.index(feature1)], features[key][all_feature_names.index(feature2)], key)



		# print(key, awstream_perf[key], vigil_perf[key], features[key][all_feature_names.index('object_size_avg')])

	# for i in range(14, len(X[0])):
	# 	fig, ax = plt.subplots()
		# ax.set_xscale('log')

		# for j in range(len(X)):

		# 	ax.scatter(X[j][i],  y[j], c='b')
		# 	# ax.text(X[j][i],  y[j], label[j])
		# 	ax.scatter(X[j][i],  y1[j], c='r')
		# 	# ax.text(X[j][i],  y1[j], label[j])
		# 	# plt.scatter(X[j][i],  y['vigil'][j], c='r')
		# 	# plt.text(X[j][i],  y['awstream'][j],label[j])

		# plt.title(all_feature_names[i])
		# plt.ylim([0, 1])

	ax0.set_xlabel(feature1)
	ax1.set_xlabel(feature2)
	ax2.set_ylabel(feature2)
	ax2.set_xlabel(feature1)
	plt.show()







	scaler = preprocessing.StandardScaler().fit(X)
	X_scaled = scaler.transform(X)  
	awstream_scores = mutual_info_regression(X,y1)
	vigil_scores = mutual_info_regression(X,y)
	rank = [sorted(vigil_scores, reverse=True).index(x) for x in vigil_scores]

	feature_indices = []
	for i in range(0, 57):
		if i in rank:
			feature_indices.append(rank.index(i))
			print(i, all_feature_names[rank.index(i)], 
				 vigil_scores[rank.index(i)], awstream_scores[rank.index(i)])



	return

if __name__ == '__main__':
	main()