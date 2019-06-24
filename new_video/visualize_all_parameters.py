import glob
import csv
import os
import numpy as np
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt


def sample(v, sample_rate):
	return v[::sample_rate]


def nonzero(orig_list):
	nonzero_list = [e for e in orig_list if e != 0]
	return nonzero_list



def compute_quantile(data):
	quantile = []
	
	quantile.append(np.percentile(data, 5))
	quantile.append(np.percentile(data, 25))
	quantile.append(np.percentile(data, 50))
	quantile.append(np.percentile(data, 75))
	quantile.append(np.percentile(data, 95))

	return quantile


def read_para(para_file):
	num_of_object = []
	object_area = []
	arrival_rate = []
	velocity = []
	total_object_area = []

	with open(para_file, 'r') as para_f:
		para_f.readline()
		for line in para_f:
			line_list = line.strip().split(',')


			if line_list[1] != '':
				num_of_object.append(int(line_list[1]))
			else:
				num_of_object.append(0)


			if line_list[2] != '':
				object_area +=  [float(x) for x in line_list[2].split(' ')]
			else:
				object_area.append(0)

			if line_list[3] != '':
				arrival_rate.append(int(line_list[1]))
			else:
				arrival_rate.append(0)

			if line_list[4] != '':
				velocity +=  [float(x) for x in line_list[4].split(' ')]
			else:
				velocity.append(0)

			if line_list[5] != '':
				total_object_area.append(float(line_list[5]))
			else:
				total_object_area.append(0)


	return num_of_object, object_area, arrival_rate, velocity, total_object_area




class Para_quantile: 
	def __init__(self): 
		self.num_of_object_quantile = {}
		self.object_area_quantile = {}
		self.arrival_rate_quantile = {}
		self.velocity_quantile = {}
		self.total_object_area_quantile = {}





def main():
	para_list = ['num_of_object', 'object_area', 'arrival_rate', 'velocity',
	'total_object_area'] 

	colors = {}
	para_quantile = Para_quantile()
	dataset_1 = ['highway','crossroad','crossroad2','crossroad3','crossroad4',
				'crossroad5','walking','driving1','driving2','KITTI_City', 
				'KITTI_Residential','KITTI_Road']
	# cn_0 = 0
	# cn_1 = 0
	# cn_2 = 0
	# cn_3 = 0
	# cn_4 = 0
	with open('stats.csv', 'w') as f:
		f.write('video name, num of objects, std, object area,std, '\
			'arrival rate, std, velocity,std, total area,std\n ')

		for para_file in sorted(glob.glob('./paras/*')):
			dataset_name = os.path.basename(para_file)\
						.replace('Video_features_', '').replace('.csv','')
			print(dataset_name)
			num_of_object, object_area, arrival_rate, velocity, \
						   total_object_area = read_para(para_file)

			para_quantile.num_of_object_quantile[dataset_name] = \
											compute_quantile(num_of_object)
			para_quantile.object_area_quantile[dataset_name] = \
											compute_quantile(object_area)
			para_quantile.arrival_rate_quantile[dataset_name] = \
											compute_quantile(arrival_rate)
			para_quantile.velocity_quantile[dataset_name] = \
											compute_quantile(velocity)			
			para_quantile.total_object_area_quantile[dataset_name] = \
											compute_quantile(total_object_area)

			f.write(dataset_name + ',' + \
				str(para_quantile.num_of_object_quantile[dataset_name][2]) + ',' + \
				str(para_quantile.object_area_quantile[dataset_name][2]) + ',' + \
				str(para_quantile.arrival_rate_quantile[dataset_name][2]) + ',' + \
				str(para_quantile.velocity_quantile[dataset_name][2]) + ',' + \
				str(para_quantile.total_object_area_quantile[dataset_name][2]) + '\n')

	fig, ax = plt.subplots(1,1, sharex=True)
	ax.boxplot(para_quantile.velocity_quantile.values(), 
		showfliers=False, patch_artist=True)
	plt.show()
	# for dataset_name in para_quantile.num_of_object_quantile.keys():
	# print(all_percent_with_object)
	# bxplot0 = ax[0].boxplot(all_quantile_size, 
	# 	showfliers=False, patch_artist=True)
	# for patch, color in zip(bxplot0['boxes'], colors):
	# 	patch.set_facecolor(color)
	# ax[0].set_ylabel('object size')
	# ax[0].set_ylim(0,0.05)		



	# 		colors.append('lightblue')


	# 	frame_rate = 10
	# 	dataset = 'KITTI-Residential'
	# 	video_index_list = [19,20,22,23,35,36,39,46,61,64,79,86,87] #[19,20,22,46,61,86,87] #
	# 	for i in video_index_list:
	# 		videoName = dataset + '_' + str(i)
	# 		f.write(videoName+',')			
	# 		file_path = '/Users/zhujunxiao/Desktop/benchmarking/KITTI/Residential/'\
	# 		'2011_09_26/2011_09_26_drive_' + format(i,'04d') + '_sync/'\
	# 		'result/para_all.csv'
	# 		comput_quantile(file_path, all_quantile_size, all_quantile_arrival, 
	# 			all_quantile_velocity, all_quantile_frame_difference, 
	# 			all_quantile_total_area, all_percent_with_object,
	# 			all_quantile_num_of_object, all_quantile_aspect_ratio,
	# 			all_quantile_object_duration, frame_rate, f)
	# 		cn_3 += 1
	# 		colors.append('limegreen')


	# 	frame_rate = 10
	# 	dataset = 'KITTI-Road'
	# 	video_index_list = [15,27,28,29,32,52,70]
	# 	for i in video_index_list:
	# 		videoName = dataset + '_' + str(i)
	# 		f.write(videoName+',')
	# 		file_path = '/Users/zhujunxiao/Desktop/benchmarking/KITTI/Road/'\
	# 		'2011_09_26/2011_09_26_drive_' + format(i,'04d') + '_sync/'\
	# 		'result/para_all.csv'
	# 		comput_quantile(file_path, all_quantile_size, all_quantile_arrival, 
	# 			all_quantile_velocity, all_quantile_frame_difference, 
	# 			all_quantile_total_area, all_percent_with_object,
	# 			all_quantile_num_of_object, all_quantile_aspect_ratio,
	# 			all_quantile_object_duration, frame_rate, f)
	# 		cn_4 += 1
	# 		colors.append('orange')



	# 	frame_rate_dict = {'walking':30,'driving_downtown':30, 'highway':25 ,
	# 	 'crossroad2': 30,'crossroad':30}

	# 	for video_index in ['walking','highway','driving_downtown',
	# 						'crossroad','crossroad2']:
	# 		dataset = 'youtube_' + video_index
	# 		frame_rate = frame_rate_dict[video_index]
	# 		f.write(dataset+',')
	# 		file_path = '/Users/zhujunxiao/Desktop/benchmarking/New_Dataset/' \
	# 		+ video_index +'/para_all.csv'
	# 		comput_quantile(file_path, all_quantile_size, all_quantile_arrival, 
	# 			all_quantile_velocity, all_quantile_frame_difference, 
	# 			all_quantile_total_area, all_percent_with_object,
	# 			all_quantile_num_of_object, all_quantile_aspect_ratio,
	# 			all_quantile_object_duration, frame_rate, f)

	# 	# frame_rate = 25
	# 	# video_index = 'highway'
	# 	# dataset = 'youtube_' + video_index
	# 	# f.write(dataset+',')
	# 	# file_path = '/Users/zhujunxiao/Desktop/benchmarking/New_Dataset/' + video_index +'/para_all.csv'
	# 	# comput_quantile(file_path, all_quantile_size, all_quantile_arrival, all_quantile_velocity, all_quantile_frame_difference, all_quantile_total_area, all_percent_with_object,all_quantile_num_of_object, frame_rate, f)

	# 	# frame_rate = 30
	# 	# video_index = 'highway'
	# 	# dataset = 'youtube_' + video_index
	# 	# f.write(dataset+',')
	# 	# file_path = '/Users/zhujunxiao/Desktop/benchmarking/New_Dataset/' + video_index +'/para_all.csv'
	# 	# comput_quantile(file_path, all_quantile_size, all_quantile_arrival, all_quantile_velocity, all_quantile_frame_difference, all_quantile_total_area, all_percent_with_object,all_quantile_num_of_object, frame_rate, f)

	# 	# frame_rate = 30
	# 	# video_index = 'crossroad'
	# 	# dataset = 'youtube_' + video_index
	# 	# f.write(dataset+',')
	# 	# file_path = '/Users/zhujunxiao/Desktop/benchmarking/New_Dataset/' + video_index +'/para_all.csv'
	# 	# comput_quantile(file_path, all_quantile_size, all_quantile_arrival, all_quantile_velocity, all_quantile_frame_difference, all_quantile_total_area, all_percent_with_object,all_quantile_num_of_object, frame_rate, f)

	# 	# frame_rate = 30
	# 	# video_index = 'crossroad2'
	# 	# dataset = 'youtube_' + video_index
	# 	# f.write(dataset+',')
	# 	# file_path = '/Users/zhujunxiao/Desktop/benchmarking/New_Dataset/' + video_index +'/para_all.csv'
	# 	# comput_quantile(file_path, all_quantile_size, all_quantile_arrival, all_quantile_velocity, all_quantile_frame_difference, all_quantile_total_area, all_percent_with_object,all_quantile_num_of_object, frame_rate, f)



	# fig, ax = plt.subplots(9,1, sharex=True)
	# print(all_percent_with_object)
	# bxplot0 = ax[0].boxplot(all_quantile_size, 
	# 	showfliers=False, patch_artist=True)
	# for patch, color in zip(bxplot0['boxes'], colors):
	# 	patch.set_facecolor(color)
	# ax[0].set_ylabel('object size')
	# ax[0].set_ylim(0,0.05)


	# bxplot1 = ax[1].boxplot(all_quantile_frame_difference,
	# 	showfliers=False, patch_artist=True)
	# for patch, color in zip(bxplot1['boxes'], colors):
	# 	patch.set_facecolor(color)
	# ax[1].set_ylabel('frame diff')
	# ax[1].set_ylim(0,75)
	
	# bxplot2 = ax[2].boxplot(all_quantile_arrival,
	# 	showfliers=False, patch_artist=True)
	# for patch, color in zip(bxplot2['boxes'], colors):
	# 	patch.set_facecolor(color)
	# ax[2].set_ylabel('arrival')
	# ax[2].set_ylim(0,10)
	

	# bxplot3 = ax[3].boxplot(all_quantile_velocity,
	# 	showfliers=False, patch_artist=True)
	# for patch, color in zip(bxplot3['boxes'], colors):
	# 	patch.set_facecolor(color)
	# ax[3].set_ylabel('velocity')
	# ax[3].set_ylim(0,1)

	# bxplot4 = ax[4].boxplot(all_quantile_total_area,
	# 	showfliers=False, patch_artist=True)
	# for patch, color in zip(bxplot4['boxes'], colors):
	# 	patch.set_facecolor(color)
	# ax[4].set_ylabel('total area')
	# ax[4].set_ylim(0,0.1)

	# ax[5].plot(range(1,len(all_quantile_size)+1), all_percent_with_object)
	# ax[5].set_ylabel('% with objects')
	# ax[5].set_ylim(0,1)	
	# ax[5].set_xlabel('Video Index')


	# bxplot5 = ax[6].boxplot(all_quantile_num_of_object,
	# 	showfliers=False, patch_artist=True)
	# for patch, color in zip(bxplot5['boxes'], colors):
	# 	patch.set_facecolor(color)
	# ax[6].set_ylabel('Number of objects')
	# ax[6].set_ylim(0,5)


	# bxplot6 = ax[7].boxplot(all_quantile_object_duration,
	# 	showfliers=False, patch_artist=True)
	# for patch, color in zip(bxplot6['boxes'], colors):
	# 	patch.set_facecolor(color)
	# ax[7].set_ylabel('Object duration')
	# ax[7].set_ylim(10,50)

	# bxplot7 = ax[8].boxplot(all_quantile_aspect_ratio,
	# 	showfliers=False, patch_artist=True)
	# for patch, color in zip(bxplot7['boxes'], colors):
	# 	patch.set_facecolor(color)
	# ax[8].set_ylabel('Aspect ratio')
	# ax[8].set_ylim(0,3)


	# ax[0].legend([bxplot0['boxes'][0], bxplot0['boxes'][1]],['0014','0060'],
	# 	bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand")
	# ax[0].legend(bxplot0['boxes'][0:len(video_index_list)],[format(i,'04d') 
	# 	for i in video_index_list], bbox_to_anchor=(0., 1.02, 1., .102), 
	# 	loc=3,ncol=len(video_index_list), mode="expand", borderaxespad=0.)
	# # ax[0].legend([bxplot0['boxes'][0], bxplot0['boxes'][cn_0],bxplot0['boxes'][cn_0+cn_1],bxplot0['boxes'][cn_0+cn_1+cn_2],bxplot0['boxes'][cn_0+cn_1+cn_2+cn_3],bxplot0['boxes'][cn_0+cn_1+cn_2+cn_3+1]],['Caltech','DukeMTMC','KITTI-City','KITTI-Residential','KITTI-Road','Walking'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=5, mode="expand", borderaxespad=0.)
	# plt.show()
	# #plt.savefig('/Users/zhujunxiao/Desktop/benchmarking/KITTI/City/2011_09_26/quantile.png')


if __name__ == '__main__':
	main()


