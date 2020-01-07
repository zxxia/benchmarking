import numpy as np
import glob
import os



def load_detection_results(dt_file):
	full_model_dt = {}
	with open(dt_file, 'r') as f:
		for line in f:
			line_list = line.strip().split(',')
			# real image index starts from 1
			img_index = int(line_list[0].split('.')[0]) #- 1
			if not line_list[1]: # no detected object
				gt_boxes_final = []
			else:
				gt_boxes_final = []
				gt_boxes = line_list[1].split(';')
				for gt_box in gt_boxes:
					# t is object type
                    box = gt_box.split(' ')
                    assert len(box) == 7,  \
                        "the length of the detection is not 7." \
                        " some data is missing"
					x = int(box[0])
					y = int(box[1])
					w = int(box[2])
					h = int(box[3])
					t = int(box[4])
                    score = float(box[5])
                    obj_id = int(box[6])
                    if t == 3 or t == 8:
					    gt_boxes_final.append([x, y, x+w, y+h, t, score, obj_id])
			full_model_dt[img_index] = gt_boxes_final

	return full_model_dt, img_index





def temporal():
	# change frame rate
    TEMPORAL_SAMPLING_LIST = [20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1]
    for sample_rate in TEMPORAL_SAMPLING_LIST:
        img_cn = 0
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        save_dt = []

        for img_index in range(start_frame, start_frame + chunk_length * frame_rate):
            dt_boxes_final = []
            current_full_model_dt = full_model_dt[img_index]
            current_gt = gt[img_index]
            resize_rate = frame_rate/standard_frame_rate
            if img_index%resize_rate >= 1:
                continue
            else:
                img_index = img_cn
                img_cn += 1


            # based on sample rate, decide whether this frame is sampled
            if img_index%sample_rate >= 1:
                # this frame is not sampled, so reuse the last saved
                # detection result
                dt_boxes_final = [box for box in save_dt]

            else:
                # this frame is sampled, so use the full model result
                dt_boxes_final = [box for box in current_full_model_dt]
                save_dt = [box for box in dt_boxes_final]

            tp[img_index], fp[img_index], fn[img_index] = \
                eval_single_image(current_gt, dt_boxes_final)

	return 


def spatial():
    # change resolution
	return

def model_pruning():
    # change different pre-trained model
	return


def main():
    dataset_name = 'highway'
    path = '/mnt/data/zhujun/dataset/Youtube/' + dataset_name + '/'
    frame_rate = 25
    # load full resolution, full model detection results as ground truth
    for resol in ['360p', '480p', '540p', '720p']:
        model = 'FasterRCNN'
        gt_file = path + resol + '/profile/' + 'updated_gt_' + model + '_COCO.csv'
        gt_bboxes[resol] = load_detection_results(gt_file)
        
    # run two modules, then show that they are independent
    # run temporal 



    # run 


        
    return

if __name__ == '__main__':
    main()