


import os
from collections import defaultdict
import glob
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn import preprocessing, metrics
import sys
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
sys.path.append('../')
from pipeline_performance_loader import Parser, initialization, read_feature
from sklearn.base import BaseEstimator, TransformerMixin
from VIF import ReduceVIF
import seaborn as sns

# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
	minmax = MinMaxScaler()
	ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
	ranks = map(lambda x: round(x,2), ranks)
	return dict(zip(names, ranks))


def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

def topK_index(data, K):
	indices = data.argsort()[-1*K:][::-1]
	return indices, data[indices]

def feature_filtering(df, filter_method='pearson'):
	# remove correlated features
	if filter_method == 'VIF':
		# filter 
		transformer = ReduceVIF(thresh=5)
		df_filtered = transformer.fit_transform(df)
		return df_filtered
	elif filter_method == 'pearson':
		# filter feateures with pearson correlation higher than a thresh
		corr_matrix = df.corr()
		correlated_features = set()
		thresh = 0.8
		for i in range(len(corr_matrix.columns)):
			for j in range(i):
				if abs(corr_matrix.iloc[i, j]) > thresh:
					colname = corr_matrix.columns[i]
					correlated_features.add(colname)
		df_filtered = df.drop(correlated_features, axis=1)
		return df_filtered

	
	else:
		print('Filter method {} does not exist.'.format(filter_method))
		return df



def select_good_features(target_perf, features, all_feature_names, NUM_OF_FEATURES=2):

	X = []
	y = []

	for key in sorted(target_perf.keys()):
		if key not in features:
			continue

		# data cleaning
		if features[key][all_feature_names.index('object_cn_median')] < 1:
			continue
		if features[key][all_feature_names.index('velocity_avg')] < 1:
			continue
		if features[key][all_feature_names.index('object_size_avg')] <= 0:
			continue

		thresh1 = 0.05		
		if np.abs(target_perf[key][1] - 0.9) > thresh1:
			continue
		X.append(features[key])
		y.append(target_perf[key][0])
	
	# Define dictionary to store our rankings
	ranks = {}
	# preprocessing: standardization, and train test split
	print('Preprocessing starts (normalization, train_test_split)......')
	scaler = preprocessing.StandardScaler().fit(X)
	X_scaled = scaler.transform(X)  
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
										test_size=0.2, random_state=0) 


	# remove correlated features, using two methods
	df = pd.DataFrame(X_train, columns=all_feature_names)
	df_filtered_pearson = feature_filtering(df)
	print(df_filtered_pearson.columns)
	df_filtered_vif = feature_filtering(df, filter_method='VIF')
	print(df_filtered_vif.columns)
	
	# visualize correlation matrix before and after filtering
	df['perf'] = y_train
	df_filtered_pearson['perf'] = y_train
	df_filtered_vif['perf'] = y_train
	
	# f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)
	cor = df.corr()
	sns.heatmap(cor, vmin=-1, vmax=1, center=0)
	plt.title('Correlation matrix before feature filtering.')
	plt.show()
	cor = df_filtered_vif.corr()
	sns.heatmap(cor, vmin=-1, vmax=1, center=0)
	plt.title('Correlation matrix after VIF filtering.')
	plt.show()
	cor = df_filtered_pearson.corr()
	sns.heatmap(cor, vmin=-1, vmax=1, center=0)
	plt.title('Correlation matrix before feature filtering.')
	plt.show()	

	
	# # feature selection method 1: pearson correlation based
	# print('Feature selection method 1: pearson correlation based')
	# cor = df_filtered.corr()
	# print(cor)
	# plt.figure(figsize=(12,10))
	
	# X = df
	# y = df_filtered['perf']
	# cor = df_filtered.corr()
	# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
	# plt.show()
	# # # feature selection method 2: random forest
	# print('Feature selection method 2: random forest based')
	# # # lr = LinearRegression(normalize=True)
	# lr = RandomForestRegressor(n_estimators=20, max_features=NUM_OF_FEATURES)
	# lr.fit(X, y)
	# rank = [np.abs(x) for x in lr.feature_importances_]
	# indicies = topK_index(np.asarray(rank), 3)
	
	# for i in indicies[0]:
	# 	print(all_feature_names[i])



	# #
	# print('Lasso method.')
	# lasso = Lasso(alpha=.05)
	# lasso.fit(X, y)
	# rank = np.abs(lasso.coef_)
	# indicies = topK_index(np.asarray(rank), 3)
	
	# for i in indicies[0]:
	# 	print(all_feature_names[i])

	# # feature selection method 3: RFE (linear regression)
	# print('RFE')
	# lr = LinearRegression(normalize=True)
	# lr.fit(X, y)
	# #stop the search when only the last feature is left
	# rfe = RFE(lr, n_features_to_select=NUM_OF_FEATURES, verbose =3 )
	# rfe.fit(X_train, y_train)
	# ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), all_feature_names, order=-1)
	# rank = np.abs(rfe.ranking_)
	# indicies = topK_index(np.asarray(rank), 3)
	
	# for i in indicies[0]:
	# 	print(all_feature_names[i])

	# print(ranks["RFE"])

	# feature selection method 4: RFE (SVR)

	# feature selection method 5: RFE (random forest)

	return


def main():
	all_feature_names, moving, video_to_delete, selected_video, glimpse_video_to_delete = initialization()
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

	awstream_perf = {}
	keys = []
	with open('../awstream/awstream_selected_video_resol_0.86.csv', 'r') as f:
		f.readline()
		for line in f:
			line_list = line.strip().split(',')
			dataset_name = line_list[0].replace('_' + 
						   line_list[0].split('_')[-1], '')
			if dataset_name in video_to_delete:
				# print(dataset_name)
				continue
			key = line_list[0]
			resol = int(line_list[2].replace('p', ''))
			f1 = float(line_list[3])
			bw = float(line_list[4])
			awstream_perf[key] = (bw, f1, resol)

	path = '/Users/zhujunxiao/Desktop/benchmarking/vldb/data/vigil/vigil_mobilenet_results_10_11/'
	pipeline = 'vigil'
	vigil_perf = Parser(pipeline, path)


	target_perf = videostorm_perf
	selected_features = select_good_features(target_perf, features, all_feature_names)


if __name__ == '__main__':
	main()
	
