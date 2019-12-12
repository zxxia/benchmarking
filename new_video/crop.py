import numpy as np
import argparse
from my_utils import load_metadata
import pdb
PATH = '/mnt/data/zhujun/dataset/Youtube/'
def overlap_percentage(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

  # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

  # compute the area of both the prediction and ground-truth
  # rectangles
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    #print(boxA, boxB)
    #print(xA, yA, xB, yB)
    return float(interArea)/boxBArea


# cropped left top point coordinates
cropped_left_top = {'crossroad3':[320, 400],
                    'crossroad4':[0, 680],
                    'crossroad5':[1320, 680],
                    'driving1':[500, 400], 'driving2':[200,300], 
                    'walking':[2840,1160], 'jp': [232,166]} 
dataset = 'jp'#args.dataset
metadata = load_metadata(PATH + dataset + '/metadata.json')
resolution = metadata['resolution'] 
coor_lt = cropped_left_top[dataset]

width = resolution[0]
height = resolution[1]
output_w = 960
output_h = 540

cropped_area = [coor_lt[0], coor_lt[1], coor_lt[0]+output_w, coor_lt[1]+output_h]
print(dataset, cropped_area)
org_file = PATH + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv'
cropped_file = PATH + 'cropped_jp/profile/gt_cropped_540p.csv'

fid_w = open(cropped_file,'w')
with open(org_file, 'r') as f:
    #fid_w.write(f.readline())
    for line in f:
        #print(line)
        line_list = line.strip().split(',')

        if len(line_list) == 1 or line_list[1] == '':
            fid.write(line_list[0] + ',\n')
        else:
            new_boxes = []
            boxes = line_list[1].split(';')
            for box_str in boxes:
                box = [int(x) for x in box_str.split(' ')]
                box[2] += box[0]
                box[3] += box[1]
                #pdb.set_trace()
                print(box)
                percentage = overlap_percentage(cropped_area, 
                                                [box[0],box[1],box[2],box[3]])
                if percentage < 0.2:
                    continue
                else:
                    box[0] = max(box[0] - coor_lt[0], 0)
                    box[1] = max(box[1] - coor_lt[1], 0)
                    box[2] = min(box[2] - coor_lt[0], output_w)
                    box[2] -= box[0]
                    box[3] = min(box[3] - coor_lt[1], output_h)
                    box[3] -= box[1]
                    new_boxes.append(' '.join([str(x) for x in box]))
            fid_w.write(line_list[0] + ',' + ';'.join(new_boxes) + '\n')

