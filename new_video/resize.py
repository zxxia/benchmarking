
import cv2
import os
from absl import app, flags 
from utils.utils import load_metadata, create_dir


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', '', 'Dataset name.')
flags.DEFINE_string('resize_resol','360p','Image resolution after resizing.')
flags.DEFINE_string('path', None, 'Data path.')

resol_dict = {'360p': (640,360),
              '480p': (854,480),
              '540p': (960,540),
              '576p': (1024,576),
              '720p': (1280, 720)
             }

def resize_video(video_in, video_out, target_size):
    cmd = "ffmpeg -y -i {} -hide_banner -vf scale={} {}".format(video_in, 
                                                                target_size, 
                                                                video_out)
    print(cmd)
    os.system(cmd)

def extract_frames(video, output_path):
    cmd = "ffmpeg -y -i {} {}%06d.jpg -hide_banner".format(video, output_path)
    print(cmd)
    os.system(cmd)

def main(argv):
    dataset = FLAGS.dataset
    path = FLAGS.path
    resol_name = FLAGS.resize_resol
    if resol_name == 'original':
        return
    target_size = resol_dict[resol_name]
    resized_path = path + dataset + '/' + resol_name + '/'
    metadata = load_metadata(path + dataset + '/metadata.json')
    resol = metadata['resolution']
    num_of_frames = metadata['frame count']
    create_dir(resized_path)
    create_dir(resized_path + 'profile/')

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
