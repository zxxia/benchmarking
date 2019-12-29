import os
import cv2
from collections import defaultdict


img_shape = (600, 400)
all_classes = ['car', 'person', 'truck',
               'bicycle', 'bus', 'motorcycle', 'no_object']


def load_annot(annot_path, separate_flag):
    # load classification labels converted directly from
    label = {}
    with open(annot_path, 'r') as f:
        for line in f:
            line_list = line.strip().split(',')
            img_index = int(line_list[0].replace('.jpg', ''))
            current_label = line_list[1].replace(' ', '').replace('"', '')

            # whether or not separate car and truck
            if separate_flag == 0 and current_label == 'car':
                current_label = 'truck'

            if len(line_list) == 2:  # 'no_object' has no area, confidence score
                label[img_index] = [current_label]
            else:

                # line_list[2] is relative box area, line_list[3] is confidence score
                # need them for future filtering
                label[img_index] = [current_label, line_list[2],
                                    line_list[3], line_list[4]]
    return label


def smooth_label(label_dict, thresh=0.7, area_thresh=0.02,
                 person_thresh=1000 / (600 * 400)):
    # if those labels appear, delete this box
    label_to_delete = ['umbrella', 'trafficlight', 'kite',
                       'skis', 'parkingmeter', 'skateboard', 'oven',
                       'dog', 'microwave', 'chair', 'tv', 'tennisracket',
                       'bird', 'firehydrant', 'toilet', 'bottle', 'sportsball',
                       'clock', 'laptop', 'stopsign', 'refrigerator',
                       'keyboard', 'mouse', 'surfboard', 'frisbee',
                       'pottedplant', 'cat', 'toothbrush', 'baseballbat',
                       'bear', 'broccoli', 'remote', 'carrot', 'sink',
                       'sheep', 'wineglass', 'handbag', 'backpack',
                       'horse', 'vase', 'knife']
    # if those labels appear, map them to 'car'
    label_to_car = ['bed']
    # if those labels appear, map them to 'truck'
    label_to_truck = ['airplane', 'boat', 'train',
                      'suitcase', 'bus', 'bench', 'book']

    new_dict = {}  # save filtered labels
    all_classes = defaultdict(int)
    for i in sorted(label_dict.keys()):
        label = label_dict[i][0]
        if len(label_dict[i]) == 1:
            # no confidence score and area information can be used
            # for filterings
            new_dict[i] = [label]
            continue
        confidence_score = label_dict[i][2]
        area = label_dict[i][1]
        box = label_dict[i][3]
        if label == 'person':
            # person will have smaller area, therefore should use
            # smaller filtering thresh (person_thresh)
            thresh2 = person_thresh
        else:
            thresh2 = area_thresh

        # if the confidence score is low or area is small
        # consider this frame as 'no_object'
        if float(confidence_score) < thresh or float(area) < thresh2:
            new_dict[i] = ['no_object']
            continue

        # map labels
        if label in label_to_delete:
            new_dict[i] = ['no_object', label_dict[i][1],
                           label_dict[i][2], label_dict[i][3]]
        elif label in label_to_car:
            new_dict[i] = ['car', label_dict[i][1],
                           label_dict[i][2], label_dict[i][3]]
        elif label in label_to_truck:
            new_dict[i] = ['truck', label_dict[i][1],
                           label_dict[i][2], label_dict[i][3]]
        else:
            new_dict[i] = [label, area,
                           confidence_score, box]

        all_classes[new_dict[i][0]] += 1
    print(all_classes.keys())

    # smooth the label by checking the label of previous frame and next frame
    for i in range(2, 31999):
        if new_dict[i-1][0] == new_dict[i+1][0]:
            if new_dict[i][0] != new_dict[i-1][0]:
                new_dict[i] = new_dict[i-1]

    return new_dict


def show_label(img_path, annot1, annot2):

    for i in range(1, 32000, 100):
        img_filename = img_path + format(i, '06d') + '.jpg'
        img = cv2. imread(img_filename)
        # if annot1[i][0] == 'person':
        if annot1[i][0] == annot2[i][0] or annot1[i][0] == 'knife':
            print(annot1[i], annot2[i])
            cv2.putText(img, ' '.join(annot1[i]), (100, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            if len(annot1[i]) != 1:
                print(annot1[i])
                x1, y1, w1, h1 = [int(x) for x in annot1[i][3].split(' ')[0:4]]
                cv2.rectangle(img, (x1, y1),
                              (x1 + w1, y1 + h1), (0, 255, 0), 3)
            cv2.putText(img, ' '.join(annot2[i]), (100, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            if len(annot2[i]) != 1:
                x2, y2, w2, h2 = [int(x) for x in annot2[i][3].split(' ')[0:4]]

                cv2.rectangle(img, (x2, y2),
                              (x2 + w2, y2 + h2), (0, 0, 255), 3)
            cv2.imshow(img_filename, img)
            cv2.waitKey(0)
    return


def easy_frame(annot, large_object_thresh=0.2):
    if annot[0] == 'no_object':
        return True
    elif float(annot[1]) >= large_object_thresh:
        return True
    else:
        return False


def eval(annot1, annot2, dataset, f):
    tp = 0
    total_cn = 0
    short_video_length = 30
    frame_rate = 30
    tp = defaultdict(int)
    total_cn = defaultdict(int)
    no_object_cn = defaultdict(int)
    easy_frame_cn = defaultdict(int)
    area = defaultdict(list)
    for i in range(1, 32000):
        # if annot1[i] == 'no_object':
        # 	continue
        if annot1[i][0] == annot2[i][0]:
            tp[i//(short_video_length*frame_rate)] += 1
        if annot1[i][0] == 'no_object':
            no_object_cn[i//(short_video_length*frame_rate)] += 1
        total_cn[i//(short_video_length*frame_rate)] += 1
        if len(annot1[i]) > 2:
            area[i//(short_video_length*frame_rate)
                 ].append(float(annot1[i][1]))
        if easy_frame(annot1[i]):
            easy_frame_cn[i//(short_video_length*frame_rate)] += 1
    # for i in sorted(tp.keys()):
    # 	if total_cn[i] == 0 or len(area[i]) == 0:
    # 		f.write(dataset + '_' + str(i) + ',0\n')
    # 	else:
    # 		f.write(dataset + '_' + str(i) + ',' + str(tp[i]/total_cn[i]) + ',' +
    # 			str(easy_frame_cn[i]/total_cn[i]) + ',' +
    # 			str(sum(area[i])/len(area[i])) + '\n')

    return


def main():
    """Need to do smoothing.

    The classification label converted from detection results is noisy.

    """
    thresh_dict = {
        'crossroad5': (0.6, 20000/(400*600)),
        'crossroad3': (0.6, 7000/(400*600)),
        'crossroad4': (0.5, 5000/(400*600)),
        'crossroad4_2': (0.5, 1000/(400*600)),
        'crossroad4_3': (0.5, 1000/(400*600)),
        'driving2': (0.6, 10000/(400*600))
    }
    person_thresh = 1000/(400*600)

    filename = '_car_truck_separate'
    # separate car and truck or not, if 1, separate, if 0, not separate
    separate_flag = 1

    # path to your converted classification label file
    path = '/Users/zhujunxiao/Desktop/benchmarking/Final_code/fast/label/'
    for dataset in ['crossroad4_3']:
        # img_path is mainly for visualization
        img_path = '/Users/zhujunxiao/Desktop/benchmarking/New_Dataset/' + dataset + '/'
        fasterrcnn_path = path + 'cropped_' + dataset + '_label_from_COCO_FasterRCNN.csv'
        fasterrcnn_label = load_annot(fasterrcnn_path, separate_flag)

        # filename of filtered and smoothed label file
        filtered_ground_truth = path + \
            'cropped_' + dataset + '_ground_truth' + filename + '.csv'

        # set two threshes for filtering, and the thresh for different videos
        # is different. Should first visualize the classification labels using
        # function show_label() to manually decide the thresholds
        thresh1 = thresh_dict[dataset][0]  # confidence score thresh
        # relative area thresh, remove small boxes
        thresh2 = thresh_dict[dataset][1]
        smoothed_fasterrcnn_label = smooth_label(
            fasterrcnn_label, thresh1, thresh2, person_thresh)

        # save smoothed labels
        with open(filtered_ground_truth, 'w') as f:
            for i in range(1, 32000):
                f.write(str(i) + ',' +
                        ','.join(smoothed_fasterrcnn_label[i]) + '\n')

        # load and smooth mobilenet labels
        annot_mobilenet_path = path + \
            'cropped_' + dataset + '_label_from_COCO_mobilenet.csv'
        mobilenet_label = load_annot(annot_mobilenet_path, separate_flag)
        smoothed_mobilenet_label = smooth_label(
            mobilenet_label, thresh1, thresh2, person_thresh)
        filtered_mobilenet_label_path = path + \
            'cropped_' + dataset + '_mobilenet' + filename + '.csv'

        # save smoothed labels
        with open(filtered_mobilenet_label_path, 'w') as f:
            for i in range(1, 32000):
                f.write(str(i) + ',' +
                        ','.join(smoothed_mobilenet_label[i]) + '\n')

        # eval(smoothed_fasterrcnn_label, smoothed_mobilenet_label, dataset, f)
        # show_label(img_path, fasterrcnn_label, mobilenet_label)
    return


if __name__ == "__main__":
    main()
