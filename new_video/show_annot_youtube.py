import cv2
from absl import app, flags
from collections import defaultdict

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'crossroad2', 'The name of youtube video.')
path = "/Users/zhujunxiao/Desktop/benchmarking/New_Dataset/"

def main(argv):

	video_index = FLAGS.dataset
	frame_to_object = defaultdict(list)
	img_path = path + video_index + '/images/'
	result_file = path + video_index + '/Parsed_ground_truth.csv'
	image_resolution_dict = {'walking': [3840,2160],
							 'driving_downtown': [3840, 2160], 
							 'highway': [1280,720],
							 'crossroad2': [1920,1080],
							 'crossroad': [1920,1080],
							 'crossroad3': [1280,720],
							 'crossroad4': [1920,1080],
							 'crossroad5': [1920,1080],
							 'driving1': [1920,1080],
							 'driving2': [1280,720]
							 }
	img_resolution = image_resolution_dict[video_index]
	empty_cn = 0
	img_cn = 0
	frame_cn = 0

	with open(result_file, 'r') as f:
		f.readline()
		for line in f:
			line_list = line.strip().split(',')
			frame_id = format(int(line_list[0]),'06d') + '.jpg'
			# frame_cn = int(line_list[0])
			# object_id = line_list[2]
			frame_to_object[frame_id].append(line_list[1:])
			# frame_to_object[frame_id].append(line_list[1] + ' ' + object_id)


	for i in range(1, 10000):
		img_name = format(i, '06d') + '.jpg'
		img = cv2. imread(img_path + img_name)
		if img_name not in frame_to_object:
			cv2.imshow(img_name, img)	

		else:
			for box in frame_to_object[img_name]:
				[x, y, w, h] = [int(x) for x in box[1:5]]
				object_id = box[0]
				label = box[5]
				cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
				cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
				cv2.putText(img, label+'_'+object_id.strip(), (x-10,y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)





			# for boxA in gt_boxes:
			# 	flag = 0
			# 	for boxB in dt_boxes_final	
			
			cv2.imshow(img_name, img)	
		cv2.waitKey(0)
	print(empty_cn)


if __name__ == '__main__':
	app.run(main)