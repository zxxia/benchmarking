import os 
import glob
from collections import defaultdict




def read_feature(feature_file):
	features = {}
	with open(feature_file) as f:
		f.readline()
		for line in f:
			line_list = line.strip().split(',')
			key = line_list[0]
			features[key] = [float(x) for x in line_list[1:]]
	return features



class Parser:
	def __init__(self, pipeline, path, video_to_delete=[], moving=[]):
		self.pipeline = pipeline
		self.path = path
		self.moving = moving
		self.video_to_delete = video_to_delete

	def VS_parser(self):
		perf = {}
		for file in glob.glob(self.path + 'videostorm_*.csv'):
			with open(file, 'r') as f:
				f.readline()
				for line in f:
					line_list = line.strip().split(',')
					f1 = float(line_list[2])
					gpu = float(line_list[1])
					perf[line_list[0]] = (gpu, f1)
		return perf
		
	def GL_parser(self):
		perf = defaultdict(list)
		moving_or_not = {}
		for file in glob.glob(self.path + 'glimpse_*.csv'):
			dataset = file.split('_')[-1].replace('.csv', '')
			if dataset in self.video_to_delete:
				continue
			with open(file, 'r') as f:
				f.readline()
				for line in f:
					line_list = line.strip().split(',')
					f1 = float(line_list[3])
					gpu = float(line_list[4])
					if dataset in self.moving:
						moving_or_not[line_list[0]] = 1
					else:
						moving_or_not[line_list[0]] = 0
					perf[line_list[0]] = (gpu, f1, float(line_list[5]))
		return perf, moving_or_not

	def AW_parser(self):
		perf = defaultdict(list)
		for file in glob.glob(self.path + 'awstream_*.csv'):
			with open(file, 'r') as f:
				f.readline()
				for line in f:
					line_list = line.strip().split(',')
					f1 = float(line_list[2])
					gpu = float(line_list[4])
					perf[line_list[0]]= (gpu, f1)	
		return perf

	def VG_parser(self):
		perf = defaultdict(list)
		for file in glob.glob(self.path + 'vigil_*.csv'):
			with open(file, 'r') as f:
				f.readline()
				for line in f:
					line_list = line.strip().split(',')
					dataset_name = line_list[0].replace('_' + \
									line_list[0].split('_')[-1], '')
					if dataset_name in self.video_to_delete:
						continue
					f1 = float(line_list[1])
					gpu = float(line_list[2])
					area = float(line_list[3])
					true_area = float(line_list[4])
					perf[line_list[0]]= (gpu, f1, area, true_area)
		return perf 

	def NS_parser(self):
		perf = {}
		with open(self.path, 'r') as f:
			f.readline()
			for line in f:
				line_list = line.strip().split(',')
				f1 = float(line_list[3])
				bw = float(line_list[4])
				gpu = float(line_list[5])
				perf[line_list[0]] = (gpu, f1, bw)

		return perf


	def MS_parser(self):
		perf = {}
		with open(self.path, 'r') as f:
			f.readline()
			for line in f:
				line_list = line.strip().split(',')
				f1 = float(line_list[3])
				gpu = float(line_list[5])
				perf[line_list[0]] = (gpu, f1, bw)		
		return perf

	def load_perf(self):
		if self.pipeline == 'videostorm':
			perf = self.VS_parser()
			return perf
		elif self.pipeline == 'glimpse':
			perf, moving_or_not = self.GL_parser()
			return perf, moving_or_not
		elif self.pipeline == 'awstream':
			perf = self.AW_parser()
			return perf
		elif self.pipeline == 'vigil':
			perf = self.VG_parser()
			return perf
		elif self.pipeline == 'noscope':
			perf = self.NS_parser()
		elif self.pipeline == 'modelselection':
			perf = self.MS_parser() 
		else:
			print('Pipeline {} does not exist!!'.format(self.pipeline))

			return None




def read_feature(feature_file):

	features = {}
	with open(feature_file) as f:
		f.readline()
		for line in f:
			line_list = line.strip().split(',')
			key = line_list[0]
			features[key] = [float(x) for x in line_list[1:]]
	return features


def initialization():
	selected_video = ['driving2_8', 'driving2_25', 'driving2_12', 'driving2_26', 
	'driving2_10', 'crossroad4_36', 'tw1_6', 'driving1_33', 'russia1_37', 
	'driving2_36', 'driving2_0', 'russia1_30', 'driving2_1', 'park_10', 
	'russia1_10', 'tw_38', 'nyc_26', 'park_14', 'park_3', 'park_8', 'driving2_21', 
	'park_2', 'park_27', 'highway_59', 'park_9', 'park_7']

	video_to_delete = ['nyc', 'russia', 'tw', 'jp', 
	'driving_downtown', 't_crossroad', 'tw1','walking','lane_split','jp_hw', 
	'highway_normal_traffic', 'tw_road','traffic', 'split']

	glimpse_video_to_delete = ['bridge','russia','lane_split','split','walking']

	

	moving = ['driving1','driving2','park','nyc', 'downtown','traffic', 'walking', 'split'] #'motorway'

	all_feature_names = [
	'object_cn_median','object_cn_avg','object_cn_mode','object_cn_var',
	'object_cn_std', 'object_cn_skewness', 'object_cn_kurtosis', 
	'object_cn_seco nd_moment', 'object_cn_percentile10', 'object_cn_percentile25', 
	'object_cn_percentile75','object_cn_percentile90', 'object_cn_iqr',
	'object_cn_entropy', 'object_size_median', 'object_size_avg', 'object_size_mode', 
	'object_size_var', 'object_size_std', 'object_size_skewness', 
	'object_size_kurtosis', 'object_size_second_moment', 'object_size_percentile10', 
	'object_size_percentile25', 'object_size_percentile75', 
	'object_size_percentile90', 'object_size_iqr', 'object_size_entropy', 
	'arrival_rate_median', 'arrival_rate_avg', 'arrival_rate_mode', 
	'arrival_rate_var', 'arrival_rate_std', 'arrival_rate_skewness',
	'arrival_rate_kurtosis', 'arrival_rate_second_moment', 
	'arrival_rate_percentile10', 'arrival_rate_percentile25', 
	'arrival_rate_percentile75', 'arrival_rate_percentile90', 'arrival_rate_iqr', 
	'arrival_rate_entropy', 'velocity_median', 'velocity_avg', 'velocity_mode', 
	'velocity_var', 'velocity_std', 'velocity_skewness', 'velocity_kurtosis', 
	'velocity_second_moment', 'velocity_percentile10', 'velocity_percentile25', 
	'velocity_percentile75', 'velocity_percentile90', 'velocity_iqr', 
	'velocity_entropy', 'total_area_median', 'total_area_avg', 'total_area_mode',  
	'total_area_var', 'total_area_std', 'total_area_skewness', 'total_area_kurtosis', 
	'total_area_second_moment', 'total_area_percentile10', 'total_area_percentile25', 
	'total_area_percentile75', 'total_area_percentile90', 'total_area_iqr', 
	'total_area_entropy', 'percentage','percentage_w_new_object','number_of_object_classes']
	 

	return all_feature_names, moving, video_to_delete, selected_video, glimpse_video_to_delete

