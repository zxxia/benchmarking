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
from pipeline_performance_loader import Parser



selected_video = ['driving2_8', 'driving2_25', 'driving2_12', 'driving2_26', 
'driving2_10', 'crossroad4_36', 'tw1_6', 'driving1_33', 'russia1_37', 
'driving2_36', 'driving2_0', 'russia1_30', 'driving2_1', 'park_10', 
'russia1_10', 'tw_38', 'nyc_26', 'park_14', 'park_3', 'park_8', 'driving2_21', 
'park_2', 'park_27', 'highway_59', 'park_9', 'park_7']

all_feature_names = [
'object_cn_median','object_cn_avg','object_cn_mode','object_cn_var',
'object_cn_std', 'object_cn_skewness', 'object_cn_kurtosis', 
'object_cn_second_moment', 'object_cn_percentile5', 'object_cn_percentile25', 
'object_cn_percentile75','object_cn_percentile95', 'object_cn_iqr',
'object_cn_entropy', 'object_size_median', 'object_size_avg', 'object_size_mode', 
'object_size_var', 'object_size_std', 'object_size_skewness', 
'object_size_kurtosis', 'object_size_second_moment', 'object_size_percentile5', 
'object_size_percentile25', 'object_size_percentile75', 
'object_size_percentile95', 'object_size_iqr', 'object_size_entropy', 
'arrival_rate_median', 'arrival_rate_avg', 'arrival_rate_mode', 
'arrival_rate_var', 'arrival_rate_std', 'arrival_rate_skewness',
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
 'total_area_entropy', 'percentage','percentage_w_new_object']


video_to_delete = ['nyc', 'russia', 'tw', 'jp', 'crossroad', 'crossroad2', 
'driving_downtown', 't_crossroad', 'tw1','walking','lane_split','jp_hw', 
'highway_normal_traffic', 'tw_road','traffic', 'split']

moving = ['driving1','driving2','park','nyc', 'driving_downtown','traffic']
 

def read_feature(feature_file):
	features = {}
	with open(feature_file) as f:
		f.readline()
		for line in f:
			line_list = line.strip().split(',')
			key = line_list[0]
			features[key] = [float(x) for x in line_list[1:]]
	return features





def topK_index(data, K):
	indices = data.argsort()[-1*K:][::-1]
	return indices, data[indices]


def main():
	feature1 = 'velocity_avg'
	feature2 = 'percentage'
	feature3 = 'object_size_avg'
	index1 = all_feature_names.index(feature1)
	index2 = all_feature_names.index(feature2)
	index3 = all_feature_names.index(feature3)
	path = '/Users/zhujunxiao/Desktop/benchmarking/vldb/data/'
	feature_file = path + 'features_all_type_width_height_filter.csv'
	features = read_feature(feature_file)

	videostorm_path = path + 'videostorm/overfitting_results/'
	videostorm_parser = Parser('videostorm', 
							  videostorm_path)
	videostorm_perf = videostorm_parser.load_perf()

	glimpse_path = path + 'glimpse/glimpse_frame_select_results/'
	glimpse_parser = Parser('glimpse', 
					  		 glimpse_path,
					  		 video_to_delete,
					  		 moving)
	glimpse_perf, moving_or_not = glimpse_parser.load_perf()

	fig = plt.figure()
	ax0 = fig.add_subplot(131)
	ax1 = fig.add_subplot(132)
	ax2 = fig.add_subplot(133)
	y = defaultdict(list)
	# ax2.set_yscale('log')



	X = []
	y = []
	for key in sorted(videostorm_perf.keys()):
		if key not in features:
			continue
		if key not in glimpse_perf:
			continue

		# data cleaning
		if features[key][all_feature_names.index('object_cn_median')] < 1:
			continue
		if features[key][all_feature_names.index('velocity_avg')] < 1:
			continue
		if features[key][all_feature_names.index('object_size_avg')] <= 0:
			continue

		thresh1 = 0.04		
		if np.abs(videostorm_perf[key][1] - 0.9) > thresh1:
			continue
		ax0.scatter(features[key][index1], videostorm_perf[key][0], c='r')
		# if moving_or_not[key] == 0:
		# 	continue



		# thresh = 0.02

		# if videostorm_perf[key][0] - glimpse_perf[key][0] > thresh:
		# # if key == 'highway':

		# # if :
		# # 	print(key)
		# 	# ax0.scatter(features[key][index1], glimpse_perf[key][0], c='b')
		# 	# ax1.scatter(features[key][index2], glimpse_perf[key][0], c='b')
		# 	ax1.text(features[key][index2], glimpse_perf[key][0], key)
		# 	ax1.text(features[key][index2], videostorm_perf[key][0], key)
		# 	# tmp1 = ax2.scatter(features[key][index1], features[key][index2], c='b')
		# 	# ax2.text(features[key][index1], features[key][index2], glimpse_perf[key][0], key)

		# 	ax21 = ax2.scatter(features[key][index2], features[key][index3], color='b')

		# 	y['compare'].append(0)
		# elif glimpse_perf[key][0] - videostorm_perf[key][0] > thresh:
		# 	# ax0.scatter(features[key][index1], videostorm_perf[key][0], c='r')
		# 	# ax1.scatter(features[key][index2], glimpse_perf[key][0], c='b')
		# 	ax1.text(features[key][index2], glimpse_perf[key][0], key)
		# 	ax1.text(features[key][index2], videostorm_perf[key][0], key)
		# 	ax22 = ax2.scatter(features[key][index2], features[key][index3], color='r')

		# 	# tmp2 = ax2.scatter(features[key][index1], features[key][index2], c='r')
		# 	# ax2.text(features[key][index1], features[key][index2], features[key][index3], key)
		# 	y['compare'].append(1)
		# else:
		# 	ax1.text(features[key][index2], glimpse_perf[key][0], key)
		# 	ax1.text(features[key][index2], videostorm_perf[key][0], key)
		# 	ax23 = ax2.scatter( features[key][index2], features[key][index3], c='k')


		# 	# ax2.text(features[key][index1], features[key][index2], features[key][index3], key)

		# 	y['compare'].append(2)
		# tmp1 = ax0.scatter(features[key][index1], videostorm_perf[key][0], c='r')

		# tmp2 = ax0.scatter(features[key][index1], glimpse_perf[key][0], c='b')

		# tmp3 = ax1.scatter(features[key][index2], videostorm_perf[key][0], c='r')
		# tmp4 = ax1.scatter(features[key][index2], glimpse_perf[key][0], c='b')
		# else:
		# 	ax1.scatter(features[key][index2], glimpse_perf[key][2], c='g')
		# 	ax1.scatter(features[key][index2], glimpse_perf[key][0], c='k')

		# ax2.text(features[key][index2], features[key][index3], key)
		X.append(features[key])
		y.append(videostorm_perf[key][0])




	# with open('tmp.csv', 'w') as f:
	# 	for key in sorted(videostorm_perf.keys()):
	# 		if key not in features or key not in glimpse_perf:
	# 			continue
	# 		f.write(key + ',' + str(videostorm_perf[key][1]) + ','+ str(videostorm_perf[key][0]) + ',')
	# 		f.write(str(glimpse_perf[key][1]) + ','+ str(glimpse_perf[key][0]) + ',')
	# 		f.write(str(features[key][index1]) + ','+ str(features[key][index2])  + ','+ str(features[key][index3])  + '\n')



	# for i in range(len(X)):
	# 	plt.scatter(X[i][index1], X[i][index2])
	# plt.xlim([0, 1])
	# plt.ylim([1, 3])
	# ax0.set_xlabel(feature1)
	# ax0.set_ylabel('gpu')
	# ax1.set_xlabel(feature2)
	# # ax1.set_ylabel(feature2)
	# ax1.set_ylabel('gpu')
	# ax2.set_xlabel(feature2)
	# ax2.set_ylabel(feature3)
	# ax2.set_ylim([0, 0.1])
	# ax2.set_zlabel(feature3)
	# ax0.legend((tmp1, tmp2), ('videostorm', 'glimpse'))
	# ax1.legend((tmp3, tmp4), ('videostorm', 'glimpse'))
	# ax2.legend((ax21, ax22, ax23), ('Glimpse better', 'VideoStorm better', 'Similar'))
	plt.show()


	scaler = preprocessing.StandardScaler().fit(X)
	X_scaled = scaler.transform(X)  
	scores = mutual_info_regression(X_scaled, y)
	rank = [sorted(scores, reverse=True).index(x) for x in scores]

	feature_indices = []
	for i in range(0, 57):
		if i in rank:
			feature_indices.append(rank.index(i))
			print(i, all_feature_names[rank.index(i)], scores[rank.index(i)])

	# selected_feature = []
	# mse = []
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
										test_size=0.2, random_state=0) 

	# lr = LinearRegression(normalize=True)
	# last_mse = 1
	# for index in feature_indices:
	# 	selected_feature.append(index)
	# 	filtered_X = [tmp[selected_feature] for tmp in X_scaled]


	# 	# X_train, X_test, y_train, y_test = train_test_split(filtered_X, y, test_size=0.2, random_state=0) 
	# # 	lr.fit(X_train, y_train)
	# # 	y_pred = lr.predict(X_test)
	# # 	new_mse = metrics.mean_squared_error(y_test, y_pred)
	# # 	print(new_mse)
	# # 	if last_mse - new_mse < 0.001:
	# # 		selected_feature.remove(index)
	# # 	last_mse = new_mse
	# # print([all_feature_names[i] for i in selected_feature])


	# # plt.plot(mse)
	# # plt.show()
	lr = LinearRegression(normalize=True)
	# lr = RandomForestRegressor(n_estimators=20, max_features=2)
	lr.fit(X_train, y_train)

	rank = [np.abs(x) for x in lr.feature_importances_]

	indicies = topK_index(np.asarray(rank), 3)
	for i in indicies[0]:
		print(i)
		print(all_feature_names[i])

	y_pred = lr.predict(X_test)





	return

if __name__ == '__main__':
	main()