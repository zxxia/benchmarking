
import cv2
import os
from absl import app, flags 
# target_size = (640,480)



FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', '', 'Dataset name.')
flags.DEFINE_string('resize_resol','360p','Image resolution after resizing.')
flags.DEFINE_integer('frame_count', 0, 'Total number of frames.')
flags.DEFINE_string('path', None, 'Data path.')

resol_dict = {'360p': (480,360),
			  '480p': (640, 480),
			  '540p': (960, 540)}



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

def main(argv):
	dataset = FLAGS.dataset
	num_of_frames = FLAGS.frame_count 
	path = FLAGS.path
	resol_name =  FLAGS.resize_resol
	target_size = resol_dict[resol_name]
	img_path = path + dataset + '/'
	resized_path = img_path + resol_name + '/'
	resol = image_resolution_dict[dataset]
	if not os.path.exists(resized_path):
		print(resized_path)
		os.mkdir(resized_path)
		os.mkdir(resized_path + 'profile/')
	else:
		print(resized_path, 'already exists!')
		return
	for i in range(1, num_of_frames):
		img = cv2.imread(img_path + format(i, '06d') + '.jpg')
		new_img = cv2.resize(img, target_size)
		cv2.imwrite(resized_path + format(i, '06d') + '.jpg', new_img)

	# update ground truth
	x_scale = target_size[0]/float(resol[0])	
	y_scale = target_size[1]/float(resol[1])
	gt_path = img_path + 'profile/updated_gt_FasterRCNN_COCO.csv'
	resized_gt_path = resized_path + 'profile/gt_'+ resol_name + '.csv'
	cn = 0
	with open(resized_gt_path, 'w') as f:
		gt_f = open(gt_path, 'r')
		for line in gt_f:
			cn += 1
			if cn > num_of_frames:
				break
			line_list = line.strip().split(',')
			if len(line_list) == 1 or line_list[1] == '':
				f.write(line_list[0] + ',\n')
			else:
				new_boxes = []
				boxes = line_list[1].split(';')
				for box_str in boxes:
					box =[int(x) for x in box_str.split(' ')]
					box[0] *= x_scale
					box[1] *= y_scale
					box[2] *= x_scale
					box[3] *= y_scale
					new_boxes.append(' '.join([str(int(x)) for x in box]))
				f.write(line_list[0] + ',' + ';'.join(new_boxes) + '\n')

	return




if __name__ == '__main__':
	app.run(main)
