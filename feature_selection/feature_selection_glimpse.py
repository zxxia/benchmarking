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
import sys
sys.path.append('../')
from pipeline_performance_loader import Parser, initialization, read_feature







def topK_index(data, K):
	indices = data.argsort()[-1*K:][::-1]
	return indices, data[indices]


def main():
	all_feature_names, moving, video_to_delete, selected_video,glimpse_video_to_delete = initialization()
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
					  		 glimpse_video_to_delete,
					  		 moving)
	glimpse_perf, moving_or_not = glimpse_parser.load_perf()

	fig = plt.figure()
	ax0 = fig.add_subplot(111)

	y = defaultdict(list)
	# ax2.set_yscale('log')


	X = []
	y = []
	for key in sorted(glimpse_perf.keys()):
		if key not in features:
			continue
		if key not in videostorm_perf:
			continue

		# data cleaning
		# if features[key][all_feature_names.index('object_cn_median')] < 1:
		# 	continue
		if features[key][all_feature_names.index('velocity_avg')] < 1:
			continue
		if features[key][all_feature_names.index('object_size_avg')] <= 0:
			continue

		thresh1 = 0.04		
		if np.abs(glimpse_perf[key][1] - 0.9) > thresh1:
			continue
		if moving_or_not[key] == 0:
			continue


			
		ax0.scatter(features[key][index2], glimpse_perf[key][0], c='r')




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
		y.append(glimpse_perf[key][0])




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
	# lr = LinearRegression(normalize=True)
	lr = RandomForestRegressor(n_estimators=20, max_features=2)
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