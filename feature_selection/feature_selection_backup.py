import os
from collections import defaultdict
import glob
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (LinearRegression, Ridge, 
								  Lasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
# from minepy import MINE
import operator

all_feature_names = ['object_size_median', 'object_size_avg', 'object_size_mode', 
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
 'total_area_entropy', 'percentage']

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
	def VS_parser(path):
		perf = {}
		for file in glob.glob(path+'videostorm_*.csv'):
			with open(file, 'r') as f:
				f.readline()
				for line in f:
					line_list = line.strip().split(',')
					f1 = float(line_list[2])
					gpu = float(line_list[1])
					perf[line_list[0]] = ((gpu, f1))


		return perf
	def GL_parser(path):
		perf = defaultdict(list)
		for file in glob.glob(path+'glimpse_*.csv'):
			with open(file, 'r') as f:
				f.readline()
				for line in f:
					line_list = line.strip().split(',')
					f1 = float(line_list[3])
					gpu = float(line_list[4])
					perf[line_list[0]] = ((gpu, f1))
		return perf

	def AW_parser(path):
		perf = defaultdict(list)
		for file in glob.glob(path+'awstream_*.csv'):
			with open(file, 'r') as f:
				f.readline()
				for line in f:
					line_list = line.strip().split(',')
					f1 = float(line_list[2])
					gpu = float(line_list[4])
					perf[line_list[0]]= ((gpu, f1))		
		return perf


	if pipeline == 'videostorm':
		perf = VS_parser(path)
	elif pipeline == 'glimpse':
		perf = GL_parser(path)
	elif pipeline == 'awstream':
		perf = AW_parser(path)
	else:
		print('Pipeline {} does not exist!!'.format(pipeline))

	return perf

def rank_to_dict(ranks, names, order=1):
	# minmax = MinMaxScaler()
	# ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
	# ranks = map(lambda x: round(x, 2), ranks)
	ranks_dict = dict(zip(names, ranks ))
	sorted_x = sorted(ranks_dict.items(), key=operator.itemgetter(1), reverse=True)

	for key in sorted_x:
		print(key)
	return 

def topK_index(data, K):
	indices = data.argsort()[-1*K:][::-1]
	return indices, data[indices]
def main():
	path = '/Users/zhujunxiao/Desktop/benchmarking/vldb/data/overfitting_results/'
	pipeline = 'videostorm'
	feature_file = '/Users/zhujunxiao/Desktop/benchmarking/vldb/data/features_all.csv'
	# data = pd.read_csv(feature_file) 
	# # print(data.head())
	# df = pd.DataFrame(data)
	# corr = df[df.columns[1:]].corr()
	# plt.imshow(corr)
	# plt.show()

	# perf_range = [0,0.2,0.4,0.6,0.8,1]

	feature1 = 'arrival_rate_avg'
	feature2 = 'velocity_avg'
	index1 = all_feature_names.index(feature1)
	index2 = all_feature_names.index(feature2)
	perf = Parser(pipeline, path)

	# perf = defaultdict(list)

	# with open('glimpse_perfect_tracking.csv', 'r') as f:
	# 	f.readline()
	# 	for line in f:
	# 		line_list = line.strip().split(',')
	# 		f1 = float(line_list[3])
	# 		gpu = float(line_list[4])
	# 		perf[line_list[0]] = ((gpu, f1))			




	features = read_feature(feature_file)
	X = []
	y = []
	for key in sorted(perf.keys()):
		if key not in features or features[key][38] == 0 or perf[key][0] > 0.9 or features[key][38] > 3:
			continue

		X.append(features[key])
		y.append(perf[key][0])



	scaler = preprocessing.StandardScaler().fit(X)
	X_scaled = scaler.transform(X)  

	# X = X_scaled.copy()
	Y = y.copy()
	for i in range(len(X)):
		plt.scatter(X[i][index1], X[i][index2])
		# plt.scatter([feature[i] for feature in X], y)
		# plt.title(all_feature_names[i])
	# plt.show()

	# promising!!!!!!!!!!!!!!!!!!!!!!!

	# estimator = LinearRegression()#SVR(kernel="linear")
	# selector = RFE(estimator, 4, step=3)
	# selector.fit(X_scaled, y)
	# tmp = selector.support_ 
	# print(selector.ranking_)
	# print(tmp)
	# for i in range(len(tmp)):
	# 	if tmp[i]:
	# 		print(all_feature_names[i])
	#####################################
 
	lr = LinearRegression(normalize=True)
	# lr.fit(X_scaled, y)
	# rank_to_dict(np.abs(lr.coef_), all_feature_names)


	# ridge = Ridge(alpha=7)
	# ridge.fit(X, Y)
	# rank_to_dict(np.abs(ridge.coef_), all_feature_names)
	 
	 
	lasso = Lasso(alpha=.05)
	# lasso.fit(X, Y)
	# rank_to_dict(np.abs(lasso.coef_), all_feature_names)
	 
	 
	# # rlasso = RandomizedLasso(alpha=0.04)
	# # rlasso.fit(X, Y)
	# # ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), all_feature_names)
	 
	# #stop the search when 5 features are left (they will get equal scores)
	rfe = RFE(lasso, n_features_to_select=3)
	rfe.fit(X,Y)
	rank_to_dict(rfe.ranking_, all_feature_names, order=-1)
	 
	# rf = RandomForestRegressor()
	# rf.fit(X,Y)
	# rank_to_dict(rf.feature_importances_, all_feature_names)
	 
	 
	# f, pval  = f_regression(X, Y, center=True)
	# ranks["Corr."] = rank_to_dict(f, all_feature_names)
	 
	# # mine = MINE()
	# # mic_scores = []
	# # for i in range(X.shape[1]):
	# # 	mine.compute_score(X[:,i], Y)
	# # 	m = mine.mic()
	# # 	mic_scores.append(m)
	 
	# # ranks["MIC"] = rank_to_dict(mic_scores, all_feature_names) 
	 
	 
	# r = {}
	# for name in all_feature_names:
	# 	r[name] = round(np.mean([ranks[method][name] 
	# 							 for method in ranks.keys()]), 2)
	 
	# methods = sorted(ranks.keys())
	# ranks["Mean"] = r
	# methods.append("Mean")
	 
	# print("\t%s" % "\t".join(methods))
	# for name in all_feature_names:
	# 	print("%s\t%s" % (name, "\t".join(map(str, 
	# 						 [ranks[method][name] for method in methods]))))

	# rf = RandomForestRegressor(n_estimators=20, max_features=2)
	# rf.fit(X_scaled, y)
	# importance = rf.feature_importances_

	# top_5_idx = np.argsort(importance)[-5:]


	# for i in list(zip([all_feature_names[x] for x in top_5_idx], importance)):
	# 	print(i)


	# print("Scores for X0, X1, X2:", map(lambda x:round (x,3),
	#                                     rf.feature_importances_))

	# indicies, values = topK_index(mutual_info_regression(X_scaled,y), 20)
	# for i in list(zip([all_feature_names[x] for x in indicies], values)):
	# 	print(i)



	return

if __name__ == '__main__':
	main()