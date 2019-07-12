
import sys
import imp
# sys.path.append('/Users/zhujunxiao/Desktop/benchmarking/code')
# gen_util = imp.load_source('utils', '/Users/zhujunxiao/Desktop/benchmarking/code/my_utils.py')
# from utils import IoU

from my_utils import IoU
import glob
import csv
import os
import time
import numpy as np
from PIL import Image
from collections import defaultdict


# path = '/Users/zhujunxiao/Desktop/benchmarking/KITTI/'
path = '/home/zhujun/video_analytics_pipelines/dataset/KITTI/'



def compute_velocity(
	current_start, current_end, frame_to_object, object_to_frame, 
	object_location, image_resolution, frame_rate):
	# no distance info from image, so define a metric to capture the velocity.
	# velocity = 1/IoU(current_box, next_box)
	velocity = {}
	last_frame = {}
	width = image_resolution[0]
	height = image_resolution[1]


	last_filename = "start"
	for frame in range(current_start,current_end + 1):
		if last_filename == "start" or frame not in frame_to_object:
			velo = 0
			velocity[frame] = velo
			last_filename = frame 
		else:
			current_boxes = frame_to_object[frame]
			objects = [int(x[0]) for x in current_boxes]
			all_velo = []
			for object_id in objects:
				key = (frame, object_id) 
				loc = object_location[key]
				# check if this object is also in the frame of 0.1 second ago
				last_filename = int(frame - 0.1*frame_rate)
				key = (last_filename, object_id)
				if key not in object_location:
					continue
				else:
					previous_loc = object_location[key]
					[x, y, w, h] = loc
					[p_x, p_y, p_w, p_h] = previous_loc
					# use IoU of the bounding boxes of this object in two 
					# consecutive frames to reflect how fast the object is moving
					# the faster the object is moving, the smaller the IoU will be
					iou = IoU([x,y,x+w,y+h],[p_x, p_y, p_w+p_x,p_h+p_y])
					if iou < 0.001:
						print('object too fast',object_id, frame)
						iou = 0.01
					# remove the influence of frame rate
					all_velo.append(1/iou)
					# all_velo.append(iou)

			if not all_velo:
				velo = ''
			else:
				velo = ' '.join(str(x) for x in all_velo) 
			velocity[frame] = velo


	return velocity



def compute_arrival_rate(
		current_start, current_end, frame_to_object, object_to_frame, frame_rate):	
	# frame_rate consecutive frames is equivalent to 1 second
	# arrival rate computes the number of new objects arrive in the following 
	# second starting from the current frame
	frame_to_event = defaultdict(int)
	for object_id in object_to_frame.keys():
		frame_id_list = object_to_frame[object_id]
		frame_id = frame_id_list[0]
		assert frame_id_list[0] == min(frame_id_list), \
			   print('Order error: {0}').format(frame_id_list)
		frame_to_event[frame_id] += 1

	arrival_rate = {}
	for current_frame in range(current_start,current_end + 1 - frame_rate):
		one_second_frames = range(current_frame, current_frame + frame_rate)
		one_second_events = 0
		for frame in one_second_frames:
			if frame not in frame_to_event:
				continue
			else:
				one_second_events += frame_to_event[frame]
		arrival_rate[current_frame] = one_second_events

	return arrival_rate



def  compute_num_of_object_per_frame(
		current_start, current_end, frame_to_object):
	num_of_objects = {}
	for frame in range(current_start,current_end + 1):
		if frame not in frame_to_object:
			num_of_objects[frame] = 0
		else:
			num_of_objects[frame] = len(frame_to_object[frame])
	return num_of_objects



def compute_object_area(current_start, current_end, frame_to_object, image_resolution):
	# for each frame, compute the avg object_area/image_size
	image_size = image_resolution[0]*image_resolution[1]
	object_area = {}
	total_object_area = {}
	object_type = {}
	dominate_object_type = {}
	for frame in range(current_start,current_end+1):			
		if frame not in frame_to_object:
			object_area[frame] = 0
			total_object_area[frame] = 0
			object_type[frame] = []
			dominate_object_type[frame] = ""
		else:
			current_boxes = frame_to_object[frame]
			current_object_size = []
			current_total_object_size = 0
			current_object_type = []
			for box in current_boxes:
				x = int(box[1])
				y = int(box[2])
				if x < 0 or y < 0 or x > image_resolution[0] or y > image_resolution[1]:
					print("box error!")
				w = int(box[3])
				h = int(box[4])
				each_object_size = float(w*h)/image_size
				current_object_size.append(each_object_size)
				# approximate the total area covered by objects 
				current_total_object_size += each_object_size
				current_object_type.append(box[5])

			object_size_str = ' '.join(str(x) for x in current_object_size)
			object_area[frame] = object_size_str
			total_object_area[frame] = current_total_object_size
			object_type[frame] = ' '.join(str(x) for x in set(current_object_type)) 
			largest_box_index = current_object_size.index(max(current_object_size))
			dominate_object_type[frame] = str(current_boxes[largest_box_index][5])

	return object_area, total_object_area, object_type, dominate_object_type


class Parameters: 
    def __init__(self): 
        self.object_area = {}
        self.total_object_area = {}
        self.object_type = {}
        self.num_of_objects = {}
        self.arrival_rate = {}
        self.velocity = {}
        self.dominate_object_type = {}




def compute_para(data, image_resolution, frame_rate):
	all_filename = data[0] 
	frame_to_object = data[1]
	object_to_frame = data[2]
	object_location = data[3]

	current_start = min(all_filename)
	current_end = max(all_filename)	

	paras = Parameters()
	paras.object_area, paras.total_object_area, paras.object_type, paras.dominate_object_type = \
		compute_object_area(current_start, current_end, frame_to_object, image_resolution)
	paras.num_of_objects = compute_num_of_object_per_frame(
		current_start, current_end, frame_to_object)
	paras.arrival_rate = compute_arrival_rate(
		current_start, current_end, frame_to_object, object_to_frame, frame_rate)
	paras.velocity = compute_velocity(
		current_start, current_end, frame_to_object, object_to_frame, 
		object_location, image_resolution, frame_rate)


	return paras


def read_annot(annot_path):
	all_filename = []
	frame_to_object = defaultdict(list)
	object_to_frame = defaultdict(list)
	object_location = {}

	with open(annot_path, 'r') as f:
		f.readline() 
		for line in f:
			# each line: (frame_id, object_id, x, y, w, h, object_type)
			line_list = line.strip().split(',')
			frame_id = int(line_list[0])
			object_id = int(line_list[1])
			if line_list[6] != 'Car':
				continue
			frame_to_object[frame_id].append(line_list[1:])
			object_to_frame[object_id].append(frame_id)
			all_filename.append(frame_id)
			key = (frame_id, object_id)
			[x, y, w, h] = [int(x) for x in line_list[2:6]]
			object_location[key] = [x, y, w, h]	
	return all_filename, frame_to_object, object_to_frame, object_location

def main():
	start_time = time.time()

	video_index_dict = {'City':[1,2,5,9,11,13,14,17,18,48,51,56,57,59,60,84,91,93],
						'Road':[15,27,28,29,32,52,70],
						'Residential':[19,20,22,23,35,36,39,46,61,64,79,86,87]}

	image_resolution = [1242, 375]
	frame_rate = 10
	for video_name in video_index_dict.keys():

		for video_index in video_index_dict[video_name]:
			current_path = path + video_name + \
			'/2011_09_26_drive_' + format(video_index,'04d') \
			+ '_sync/'
			annot_path =  current_path + 'Parsed_ground_truth.csv'
			output_folder = './paras'
			print(annot_path)
			if not os.path.exists(output_folder):
				os.makedirs(output_folder)

			# read annotations from ground truth file
			data = read_annot(annot_path)
			paras = compute_para(data, image_resolution, frame_rate)

			all_filename = data[0] 
			current_start = min(all_filename)
			current_end = max(all_filename)				
			para_file = output_folder + '/Video_features_KITTI_' + video_name +\
					    '_' + str(video_index) + '.csv'
			with open(para_file, 'w') as f:
				f.write('frame_id, num_of_object, object_area, arrival_rate,'\
					'velocity, total_object_area, num_of_object_type\n')
				for frame_id in range(current_start, current_end + 1 - frame_rate):
					f.write(str(frame_id) + ',')
					f.write(str(paras.num_of_objects[frame_id]) + ',')
					f.write(str(paras.object_area[frame_id]) + ',')
					f.write(str(paras.arrival_rate[frame_id]) + ',')
					f.write(str(paras.velocity[frame_id]) + ',')
					f.write(str(paras.total_object_area[frame_id]) + ',')
					f.write(str(paras.object_type[frame_id]) + '\n')
			print(time.time()-start_time)

if __name__ == '__main__':
	main()



				
