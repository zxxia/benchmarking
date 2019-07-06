
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
              '480p': (640,480),
              '540p': (960,540),
             }



image_resolution_dict = {'walking': [3840,2160],
                         'driving_downtown': [3840, 2160], 
                         'highway': [1280,720],
                         'crossroad2': [1920,1080],
                         'crossroad': [1920,1080],
                         'crossroad3': [1280,720],
                         'crossroad4': [1920,1080],
                         'crossroad5': [1920,1080],
                         'driving1': [1920,1080],
                         'driving2': [1280,720],
                         'traffic': [1280,720],
                         'highway_no_traffic': [1280,720],
                         'highway_normal_traffic': [1280,720],
                         'street_racing': [1280,720],
                         'motor': [1280,720],
                         'reckless_driving': [1280,720],
                        }

def resize_video(video_in, video_out, target_size):
    cmd = "ffmpeg -i {} -vf scale={} {}".format(video_in, 
                                                target_size, 
                                                video_out)
    print(cmd)
    os.system(cmd)

def extract_frames(video, output_path):
    cmd = "ffmpeg -i {} {}%06d.jpg".format(video, output_path)
    print(cmd)
    os.system(cmd)

def create_dir(path):
    if not os.path.exists(path):
        print('create path ', path)
        os.mkdir(path)
    else:
        print(path, 'already exists!')


def main(argv):
    dataset = FLAGS.dataset
    num_of_frames = FLAGS.frame_count 
    path = FLAGS.path
    resol_name = FLAGS.resize_resol
    target_size = resol_dict[resol_name]
    resized_path = path + dataset + '/' + resol_name + '/'
    resol = image_resolution_dict[dataset]
    create_dir(resized_path)
    create_dir(resized_path + 'profile/')

    # Old method to resize videos and frames 
    # img_path = path + dataset + '/images/'
    # Directly resize frames - Deprecated
    # for i in range(1, num_of_frames + 1):
    # in_img = img_path + format(i, '06d') + '.jpg'
    # out_img = resized_path + format(i, '06d') + '.jpg'
    # resize_img(in_img, out_img, 
    # str(target_size[0]) + ':'+str(target_size[1]))
    # img = cv2.imread(img_path + format(i, '06d') + '.jpg')
    # new_img = cv2.resize(img, target_size)
    # cv2.imwrite(resized_path + format(i, '06d') + '.jpg', new_img)
    
    # resize the video
    orig_video = path + dataset + '/' + dataset + '.mp4'
    resized_video = resized_path + dataset + '_' + resol_name + '.mp4'
    resize_video(orig_video, resized_video, str(target_size[0]) + ':' + str(target_size[1]))
    # extract frames from the resized videos
    extract_frames(resized_video, resized_path)
    # update ground truth
    x_scale = target_size[0]/float(resol[0])        
    y_scale = target_size[1]/float(resol[1])
    gt_path = path + dataset + '/profile/updated_gt_FasterRCNN_COCO.csv'
    resized_gt_path = resized_path + 'profile/gt_'+resol_name+ '.csv'
    cn = 0
    print(resized_gt_path)
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





if __name__ == '__main__':
    app.run(main)
