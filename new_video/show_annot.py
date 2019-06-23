import cv2
from absl import app, flags
from collections import defaultdict

# path = "/Users/zhujunxiao/Desktop/test_gt/"
path = "/Users/zhujunxiao/Desktop/benchmarking/New_Dataset/highway/"

def show(frame_id, img_path, frame_to_object):
	img_name = format(frame_id, '06d') + '.jpg'
	img = cv2. imread(img_path + img_name)
	print(img_path + img_name)
	if frame_id not in frame_to_object:
		cv2.imshow(img_name, img)	

	else:
		for box in frame_to_object[frame_id]:
			[x, y, w, h] = [int(x) for x in box[0:4]]
			# print(x, y, w, h)
			label = str(box[4])
			# object_id = str(box[5])
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
			cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)





		# for boxA in gt_boxes:
		# 	flag = 0
		# 	for boxB in dt_boxes_final	
		
		cv2.imshow(img_name, img)	
	cv2.waitKey(0)
	return

def main(argv):

	frame_to_object = defaultdict(list)
	img_path = path  + 'images/'
	result_file = path +  '/profile/gt_FasterRCNN_COCO.csv'
	img_resolution = (1280,720) #(1920, 1080)#,
	empty_cn = 0
	img_cn = 0
	frame_cn = 0

	with open(result_file, 'r') as f:
		for line in f:
			line_list = line.strip().split(',')
			frame_id = line_list[0]
			# frame_cn = int(line_list[0])
			# object_id = line_list[2]
			if line_list[1] == '':
				continue
			else:
				gt_str = line_list[1].split(';')
				for box_str in gt_str:
					box = box_str.split(' ')
					frame_to_object[frame_id].append(box)
			# frame_to_object[frame_id].append(line_list[1] + ' ' + object_id)


	for i in range(1, 1000):
		show(i, img_path, frame_to_object)
		# img_name = format(i, '06d') + '.jpg'
		# img = cv2. imread(img_path + img_name)
		# print(img_path + img_name)
		# if img_name not in frame_to_object:
		# 	cv2.imshow(img_name, img)	

		# else:
		# 	for box in frame_to_object[img_name]:
		# 		[x, y, w, h] = [int(x) for x in box[0:4]]
		# 		print(x, y, w, h)
		# 		label = box[4]
		# 		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
		# 		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
		# 		cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)





		# 	# for boxA in gt_boxes:
		# 	# 	flag = 0
		# 	# 	for boxB in dt_boxes_final	
			
		# 	cv2.imshow(img_name, img)	
		cv2.waitKey(0)
	print(empty_cn)


if __name__ == '__main__':
	app.run(main)