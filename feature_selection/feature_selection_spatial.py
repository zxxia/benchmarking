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
 'total_area_entropy', 'percentage','percentage w/ new objects']


video_to_delete = ['nyc', 'russia',  'crossroad2', 'driving_downtown',
'tw_road','tw_under_bridge','tw1','lane_split','tw','crossroad3']
# video_to_delete = ['lane_split', 'russia', 'tw_under_bridge','highway_normal_traffic']

def read_feature(feature_file):
	features = {}
	with open(feature_file) as f:
		f.readline()
		for line in f:
			line_list = line.strip().split(',')
			key = line_list[0]
			features[key] = [float(x) for x in line_list[1:]]
	return features



def AW_parser(path):
	perf = defaultdict(list)
	for file in glob.glob(path+'awstream_*.csv'):
		with open(file, 'r') as f:
			f.readline()
			for line in f:
				line_list = line.strip().split(',')
				dataset_name = line_list[0].replace('_' + line_list[0].split('_')[-1], '')
				if dataset_name in video_to_delete:
					# print(dataset_name)
					continue

				resol = int(line_list[1].replace('p', ''))
				f1 = float(line_list[2])
				gpu = float(line_list[4])
				perf[line_list[0]]= (gpu, f1, resol)	
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
	feature1 = 'object_size_avg'
	feature2 = 'object_size_percentile25'
	feature3 = 'percentage'
	index1 = all_feature_names.index(feature1)
	index2 = all_feature_names.index(feature2)
	index3 = all_feature_names.index(feature3)

	feature_file = '/Users/zhujunxiao/Desktop/benchmarking/vldb/data/features_all_type_width_height_filter.csv'
	features = read_feature(feature_file)

	path = '/Users/zhujunxiao/Desktop/benchmarking/vldb/data/awstream/spatial_overfitting_results_09_28/'
	pipeline = 'awstream'
	awstream_perf = AW_parser(path)

	path = '/Users/zhujunxiao/Desktop/benchmarking/vldb/data/awstream/spatial_overfitting_profile_09_28/'
	total_object_cn = profile_parser(path)


	y = []

	X = []
	label = {}
	cn = 0
	fig, ax = plt.subplots()
	# ax.set_xscale('log')
	for key in sorted(awstream_perf.keys()):
		if key not in features:
			continue

		if features[key][all_feature_names.index('velocity_avg')] < 1 or features[key][all_feature_names.index('object_size_avg')] <= 0:
			continue
		# if the total_object_cn is small, the result is not robust. one object could easily baise the f1 score
		if  total_object_cn[key] < 300:
			continue
		# if features[key][all_feature_names.index('percentage')] < 0.6:
		# 	continue


		# thresh1 = 0.05

		# if np.abs(awstream_perf[key][1] - 0.9) > thresh1:
		# 	continue

		X.append(features[key])
		label[cn] = key
		cn += 1
		y.append(awstream_perf[key][0])
		ax.scatter(features[key][all_feature_names.index(feature1)], awstream_perf[key][2], c='b')
		ax.text(features[key][all_feature_names.index(feature1)], awstream_perf[key][2], key)
	plt.xlim([0,0.05])
	plt.show()




	# scaler = preprocessing.StandardScaler().fit(X)
	# X_scaled = scaler.transform(X)  
	# scores = mutual_info_regression(X,y)
	# rank = [sorted(scores, reverse=True).index(x) for x in scores]

	# feature_indices = []
	# for i in range(0, 57):
	# 	if i in rank:
	# 		feature_indices.append(rank.index(i))
	# 		print(i, all_feature_names[rank.index(i)], scores[rank.index(i)])



	return

if __name__ == '__main__':
	main()