import csv
import numpy as np
from utils.utils import write_json_file,write_pickle_file,read_pickle_file
"""Profiler module main process_feature_performance. """
def read_feature(feature_output_filename,interested_feature_list):
	""" 
		output a dict that maps clip id to a dict of interested features
	"""
	feature_dic={}
	with open(feature_output_filename, 'r') as f:
		reader = csv.reader(f)
		header = next(reader)
		interested_feature_list_id={}
		for interested_feature in interested_feature_list:
			for i, head in enumerate(header):
				if head==interested_feature:
					interested_feature_list_id[interested_feature]=i
		for i, row in enumerate(reader):
			per_frame_feature_dic={}
			for interested_feature in interested_feature_list:
				per_frame_feature_dic[interested_feature]=float(row[interested_feature_list_id[interested_feature]])
			feature_dic[i]=per_frame_feature_dic
	return feature_dic
def read_performance(pipeline_performance_filename,interested_performance_list,oracle_model,oracle_temporal, oracle_spatial, model_flag, temporal_flag, spatial_flag):
	temporal_performance_dict={}
	#TODO: support spatial_performance_dict
	spatial_performance_dict={}
	model_performance_dict={}

	with open(pipeline_performance_filename, 'r') as f:
		reader = csv.reader(f)
		header = next(reader)
		rev_map={} #a dict that maps an attribute to a column id
		for i, head in enumerate(header):
			rev_map[head]=i
		# print(rev_map)
		for i, row in enumerate(reader):
			cur_clip_id=int(row[rev_map['video_name']].split('_')[-1])
			# print(cur_frame_id)
			f1=float(row[rev_map['f1']])
			if str(row[rev_map.get('gpu time')])!='NA':
				gpu_time=float(row[rev_map['gpu time']])
			else:
				gpu_time=None
			if str(row[rev_map.get('bandwidth')])!='NA':
				bandwidth=float(row[rev_map['bandwidth']])
			else:
				bandwidth=None
			# print(cur_clip_id)
			# print(bandwidth)
			# print(gpu_time)
			#TODO: make it to be pipeline invariant 
			model=str(row[rev_map['model']])
			spatial=str(row[rev_map['spatial']])
			temporal=str(row[rev_map['temporal']])
			if temporal_flag and (model==oracle_model or model=='NA') and (spatial==oracle_spatial or spatial=='NA'):
				if temporal_performance_dict.get(temporal) is None:
					temporal_performance_dict[temporal]={}
				temporal_performance_dict[temporal][cur_clip_id]={'f1': f1, 'gpu time':gpu_time, 'bandwidth':bandwidth}
			if spatial_flag and (model==oracle_model or model=='NA') and (temporal==oracle_temporal or temporal=='NA'):
				if spatial_performance_dict.get(spatial) is None:
					spatial_performance_dict[spatial]={}
				spatial_performance_dict[spatial][cur_clip_id]={'f1': f1, 'gpu time':gpu_time, 'bandwidth':bandwidth}
			if model_flag and (spatial==oracle_spatial or spatial=='NA') and (temporal==oracle_temporal or temporal=='NA'):
				if model_performance_dict.get(model) is None:
					model_performance_dict[model]={}
				model_performance_dict[model][cur_clip_id]={'f1': f1, 'gpu time':gpu_time, 'bandwidth':bandwidth}
		print(spatial_performance_dict)
		print(spatial_flag)
		# print(model_performance_dict)
	return temporal_performance_dict,spatial_performance_dict, model_performance_dict

def value_to_bin(value, bucket_info):
	MIN, MAX, NUM_BIN=bucket_info
	#TODO: 2 ways, need to be disccused, 
	# 1:
	# if (value<MIN or value>MAX):
	# 	return None
	# 2:
	if value<=MIN:
		return 0
	if value>=MAX:
		return NUM_BIN-1

	BIN_SIZE=(MAX-MIN)/NUM_BIN
	return int((value-MIN)/(BIN_SIZE)) #truncate

def bucketize_feature(feature_dict,interested_feature_bucket_info_dict):
	bucketized_feature_dict={}
	for i, per_frame_feature_dict in enumerate(feature_dict):
		per_frame_bucketized_feature_dict={}
		for feature, value in feature_dict[i].items():
			per_frame_bucketized_feature_dict[feature]=value_to_bin(value, interested_feature_bucket_info_dict[feature])
		bucketized_feature_dict[i]=per_frame_bucketized_feature_dict
	return bucketized_feature_dict

def write_profile(bucketized_feature_dict, performance_dict, feature_list,interested_performance_list, pruning_profile_filename,pruning_profile_pickle_filename):
	if (performance_dict is None) or  (performance_dict is {}): return  None
	# print(performance_dict)
	profile_dict={}
	statistic_profile_dict={}
	for_show_statistic_profile_dict={}# just for showing purposes(saved to json)
	for config, all_clip_dic in performance_dict.items():
		profile_dict[config]={}
		for clipid, performance in all_clip_dic.items():
			feature_bin=[]
			for feature in feature_list:
				feature_bin.append(bucketized_feature_dict[clipid][feature])
			
			# print(feature_bin)
			tmp_performance=[]
			#eg. interested_performance_list=['f1', 'gpu time', 'bandwidth']
			for interested_performance in interested_performance_list:
				# print(interested_performance)
				# print(performance)
				tmp_performance.append(performance[interested_performance])
			if profile_dict[config].get(tuple(feature_bin)) is None:
				profile_dict[config][tuple(feature_bin)]=[]
			profile_dict[config][tuple(feature_bin)].append(tuple(tmp_performance))
		for feature_bin, performance_list in profile_dict[config].items():
			# print(config, feature_bin)
			tmp=np.stack(performance_list,axis=0)
			# print(tmp)
			# exit(0)
			# print(tmp.shape)
			#do statistic on each performance
			statistic_per_config_per_feature_bin_performance_dict={}
			for_show_statistic_per_config_per_feature_bin_performance_dict={}
			for i, interested_performance in enumerate(interested_performance_list):
				if tmp[0, i] is None:
					mean=None
					var=None
				else:
					mean=np.mean(tmp[:,i])
					var=np.var(tmp[:,i])
				
				statistic_per_config_per_feature_bin_performance_dict[interested_performance]=(mean, var)
				for_show_statistic_per_config_per_feature_bin_performance_dict[interested_performance]={'mean':mean, 'var':var}
				# print(tmp[:,i])
				# print('mean', mean)
				# print('var', var)
			if statistic_profile_dict.get(config) is None:
				statistic_profile_dict[config]={}
			if for_show_statistic_profile_dict.get(config) is None:
				for_show_statistic_profile_dict[config]=[]
			feature_bin_dic_tmp={}
			for feature, bin_num in zip(feature_list,feature_bin):
				feature_bin_dic_tmp[feature]=bin_num

			statistic_profile_dict[config][feature_bin]=statistic_per_config_per_feature_bin_performance_dict
	
			
			# print(feature_bin_str)
			for_show_statistic_profile_dict[config].append(feature_bin_dic_tmp)
			for_show_statistic_profile_dict[config].append(for_show_statistic_per_config_per_feature_bin_performance_dict)
	print(statistic_profile_dict)
	write_json_file(pruning_profile_filename, for_show_statistic_profile_dict)
	write_pickle_file(pruning_profile_pickle_filename,statistic_profile_dict)
	return statistic_profile_dict


def process_feature_performance(args):
	interested_feature_list=args.interested_feature_list 
	feature_output_filename=args.feature_output_filename
	feature_dict=read_feature(feature_output_filename,interested_feature_list)
	pipeline_performance_filename=args.pipeline_performance_filename
	interested_performance_list=args.interested_performance_list 
	temporal_performance_dict,spatial_performance_dict, model_performance_dict=read_performance(pipeline_performance_filename, interested_performance_list,args.oracle_model,args.oracle_temporal, args.oracle_spatial,args.model_flag, args.temporal_flag, args.spatial_flag)
	interested_feature_bucket_info_dict=eval(args.interested_feature_bucket_info_dict)
	bucketized_feature_dict=bucketize_feature(feature_dict,interested_feature_bucket_info_dict)
	write_pickle_file('interested_feature_bucket_info_dict.bin', interested_feature_bucket_info_dict)
	write_pickle_file('pipeline_profiler_args.bin', args)
	#temporal
	write_profile(bucketized_feature_dict, temporal_performance_dict, args.temporal_feature_list,interested_performance_list, args.temporal_pruning_profile_filename,'temporal_pruning_profile.bin' )
	#spatial
	write_profile(bucketized_feature_dict, spatial_performance_dict, args.spatial_feature_list,interested_performance_list, args.spatial_pruning_profile_filename,'spatial_pruning_profile.bin' )
	#model
	write_profile(bucketized_feature_dict, model_performance_dict, args.model_feature_list,interested_performance_list, args.model_pruning_profile_filename,'model_pruning_profile.bin')
	# print(read_pickle_file('model_pruning_profile.bin'))
	# print(read_pickle_file('temporal_pruning_profile.bin'))
	# print(interested_feature_bucket_info_dict)

	# print(feature_dict)
	return 



