from infer_object_id_youtube import smooth, read_annot, tag_object
import cv2
from utils.model_utils import load_full_model_detection

img_path = '/mnt/data/zhujun/dataset/Youtube/motorway/720p/'
# fastrcnn_annot_file = '/mnt/data/zhujun/dataset/Youtube/motorway/720p/profile/gt_FasterRCNN_COCO.csv'
annot_file = '/home/zxxia/benchmarking/Vigil/mobilenet_test/motorway/gt_mobilenet_720p.csv'
output_file = 'test_smooth_dir/Parsed_gt_mobilenet.csv'
updated_file = 'test_smooth_dir/Updated_gt_mobilenet.csv'
print(annot_file)

frame_ids, frame_to_obj = read_annot(annot_file, 720//20)

frame_to_obj_before_smooth = frame_to_obj.copy()
frame_to_obj = smooth(frame_to_obj)
tag_object(sorted(frame_to_obj.keys()), frame_to_obj, output_file, updated_file)

tagged_frame_to_obj, _ = load_full_model_detection(updated_file)

def visualize1(img, frame_id, boxes, color=(0, 0, 0), mobilenet_flag=False):
    for box in boxes:
        # print(box)
        [x, y, xmax, ymax, t, score, obj_id] = box
        if mobilenet_flag:
            cv2.rectangle(img, (x, y), (xmax, ymax), color, 3)
        else:
            cv2.rectangle(img, (x, y), (xmax, ymax), color, 2)

        # cv2.putText(img, str(t), (x-10, y-10),
        #             cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        if mobilenet_flag:
            # cv2.putText(img, '{:.3f}'.format(score), (x+80, ymax+10),
                        # cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            cv2.putText(img, '{:d}'.format(obj_id), (x+10, ymax+10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            # cv2.putText(img, 'smoothed', (x, ymax+10),
            #             cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        else:
            pass
            # cv2.putText(img, '{:.3f}'.format(score), (x+80, y-10),
                        # cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            # cv2.putText(img, str(obj_id), (x, y-10),
            #             cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)


def visualize(img, frame_id, boxes, color=(0, 0, 0), mobilenet_flag=False):
    for box in boxes:
        # print(box)
        [x, y, w, h, t, score] = box
        if mobilenet_flag:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

        # cv2.putText(img, str(t), (x-10, y-10),
        #             cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        if mobilenet_flag:
            cv2.putText(img, '{:.3f}'.format(score), (x+80, y + h+10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            cv2.putText(img, 'smoothed', (x, y+h+10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        else:
            cv2.putText(img, '{:.3f}'.format(score), (x+80, y-10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            # cv2.putText(img, str(obj_id), (x, y-10),
            #             cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)


for i in range(1, 10000):
# for i in range(150, 24*10):
    img_name = format(i, '06d') + '.jpg'
    img = cv2. imread(img_path + img_name)
    # print('before smooth {}, after smooth {}'.format(len(frame_to_obj_before_smooth[i]), len(frame_to_obj[i])))
    # visualize(img, img_path, frame_to_obj[i], (0, 255, 0), True)
    # visualize(img, img_path, frame_to_obj_before_smooth[i], (255, 0, 0))
    visualize1(img, img_path, tagged_frame_to_obj[i], (255, 0, 0), True)
    cv2.imshow(img_name, img)
    cv2.moveWindow(img_name, 200, 200)
    if i > 10:
        cv2.destroyWindow(format(i-10, '06d') + '.jpg')
    c = cv2.waitKey(0)
    if c & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
