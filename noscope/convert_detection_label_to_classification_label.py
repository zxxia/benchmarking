from collections import defaultdict


def label_mapping(label_map_path):
    label_map = {}
    with open(label_map_path, 'r') as f:
        line = f.readline()
        while line:
            if 'id' in line:
                ID = int(line.strip().split(':')[1].strip())
                line = f.readline()
                label = line.strip().split(':')[1]
                label_map[ID] = label
                line = f.readline()
            else:
                line = f.readline()
    # for node in graph_def.node:
    return label_map


# output of our detection model is numbers, we need to map them to
# real labels based on mscoco_label_map.pbtxt
filepath = '/home/zhujunxiao/video_analytics_pipelines/models/'\
    'research/object_detection/data/mscoco_label_map.pbtxt'
label_map = label_mapping(filepath)
img_shape = (600, 400)


# map detection results of all 4 models to classification label
for model in ['mobilenet', 'inception', 'FasterRCNN', 'FasterRCNN50']:
    for dataset in ['cropped_driving1']:
        path = ('/mnt/data/zhujun/dataset/Youtube/' + dataset +
                '/360p/profile/updated_gt_' + model + '_COCO_no_filter.csv')

        # classification label file path
        result_file = open('/home/zhujunxiao/video_analytics_pipelines/final_code/fast/label/'
                           + dataset + '_label_from_COCO_' + model + '.csv', 'w')
        print(model, dataset)
        gt_dict = defaultdict(list)

        # load the bbox for each frame
        with open(path, 'r') as f:
            f.readline()
            for line in f:
                line_list = line.strip().split(',')
                frame_id = int(line_list[0])
                if line_list[1] == "":
                    continue
                for box in line_list[1].split(';'):
                    box_list = box.split(' ')
                    if len(box_list) != 7:
                        print(line)
                        print(box_list)
                    label = label_map[int(box_list[4])]
                    area = float(box_list[2])*float(box_list[3])
                    confidence = float(box_list[5])

                    # remove very large box
                    if area > 80000:
                        continue

                    # sometimes there is a constantly wrong box which messes up
                    # the performance, then remove it
                    # if float(box_list[0]) < 100 and float(box_list[1]) < 100:
                    # 	continue
                    gt_dict[frame_id].append(
                        (label, area, confidence, ' '.join([str(x) for x in box_list])))

        for i in range(1, frame_id):
            if i not in gt_dict:
                # if no box, then label no_object
                result_file.write(str(i) + ', no_object\n')
            else:
                # use the label of largest box as the label for this frame
                labels = gt_dict[i]
                area = [x for (_, x, _, _) in labels]
                confidence = [x for (_, _, x, _) in labels]
                box = [x for (_, _, _, x) in labels]
                index = area.index(max(area))
                final_label = labels[index]
                conf = confidence[index]
                box_str = box[index]
                # important! save relative box area
                result_file.write(str(i) + ',' + final_label[0] + ',' +
                                  str(max(area)/(img_shape[0]*img_shape[1])) + ',' + str(conf) + ',' + box_str + '\n')
