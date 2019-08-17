from collections import defaultdict
from utils.model_utils import load_full_model_detection_new, eval_single_image
from utils.utils import load_metadata, compute_f1
import numpy as np
import os
import subprocess
import pdb

PATH = '/mnt/data/zhujun/dataset/Youtube/'
DATASET_LIST = ['motorway']
ALPHA = 0.2
BETA = 0.4
OFFSET = 2*60# + 30
TEST_LENGTH = 30
RESOLUTION_LIST = ['480p', '540p', '720p']
RESOLUTION_DICT = {'360p': (640,360),
              '480p': (854,480),
              '540p': (960,540),
              '576p': (1024,576),
              '720p': (1280,720),
              '1080p': (1920,1080),
              'original': (1280, 720)
             }
DEBUG = False
#DEBUG = True

def debug_print(msg):
    if DEBUG:
        print(msg)

def crop_image(img_in, xmin, ymin, xmax, ymax, w_out, h_out, img_out):
    cmd= ['ffmpeg',  '-y', '-hide_banner', '-i', img_in, '-vf',
          'crop={}:{}:{}:{},pad={}:{}:0:0:black'.format(xmax-xmin, ymax-ymin,
          xmin, ymin, w_out, h_out), img_out]

    #print(cmd)
    subprocess.run(cmd)

    #pdb.set_trace()
    #cmd = ['ffmpeg', '-y', '-vcodec', 'libx264',
    #       '-crf', 25, '-pix_fmt', 'yuv420p', '-hide_banner', 'crop.mp4']

    #size = os.path.getsize('crop.mp4')
    #os.remove(img_out)
    #return size

def compress_images_to_video_v1(path: str, start_frame: int, nb_frames: int,
                             frame_rate, resolution,
                             quality: int, output_name: str):
    '''
    Compress a set of frames to a video.
    start frame: input image frame start index
    '''
    cmd = ['ffmpeg', '-r', str(frame_rate), '-f', 'image2', '-s', str(resolution),
           '-start_number', str(start_frame), '-i', '{}%06d.jpg'.format(path),
           '-vframes', str(nb_frames), '-vcodec', 'libx264', '-crf', str(quality),
           '-pix_fmt', 'yuv420p', '-hide_banner', output_name]
    subprocess.run(cmd)#, stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip()

    video_size = os.path.getsize(output_name)
    os.remove(output_name)
    return video_size



def compress_images_to_video(list_file: str, frame_rate: str, resolution: str,
                             quality: int, output_name: str):
    '''
    Compress a set of frames to a video.
    '''
    cmd = ['ffmpeg', '-y', '-r', str(frame_rate), '-f', 'concat', '-safe', '0',
           '-i', list_file, '-s', str(resolution), '-vcodec', 'libx264',
           '-crf', str(quality), '-pix_fmt', 'yuv420p', '-hide_banner',
           output_name]
    subprocess.run(cmd)

def compute_video_size(img_path, start, end, target_frame_rate, frame_rate,
                       image_resolution):
    #pdb.set_trace()
    sample_rate = frame_rate/target_frame_rate
    # Create a tmp list file contains all the selected iamges
    tmp_list_file = 'list.txt'
    with open(tmp_list_file, 'w') as f:
        for img_index in range(start, end + 1):
            # based on sample rate, decide whether this frame is sampled
            if img_index%sample_rate >= 1:
                continue
            else:
                line = 'file \'{}/{:06}.jpg\'\n'.format(img_path, img_index)
                f.write(line)

    frame_size = str(image_resolution[0]) + 'x' + str(image_resolution[1])
    compress_images_to_video(tmp_list_file, target_frame_rate, frame_size,
                             25, 'tmp.mp4')
    video_size = os.path.getsize("tmp.mp4")
    os.remove('tmp.mp4')
    #os.remove(tmp_list_file)
    print('target frame rate={}, target image resolution={}. video size={}'.format(target_frame_rate, image_resolution, video_size))
    return video_size


def scale(box, in_resol, out_resol):
    """
    box: [x, y, w, h]
    in_resl: (width, height)
    out_resl: (width, height)
    """
    x_scale = out_resol[0]/in_resol[0]
    y_scale = out_resol[1]/in_resol[1]
    box[0] = int(box[0] * x_scale)
    box[1] = int(box[1] * y_scale)
    box[2] = int(box[2] * x_scale)
    box[3] = int(box[3] * y_scale)
    return box

def convert_box_coordinate(box):
    """
    box: x, y, w, h
    return: [xmin, ymin, xmax, ymax]
    """
    xmin = box[0]
    ymin = box[1]
    xmax = xmin + box[2]
    ymax = ymin + box[3]
    return [xmin, ymin, xmax, ymax]

def overlap_percentage(boxA, boxB):
    """
    boxA, boxB: [xmin, ymin, xmax, ymax]
    """
    #boxA = convert_box_coordinate(boxA)
    #boxB = convert_box_coordinate(boxB)
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


def main():
    for dataset in DATASET_LIST:
        metadata = load_metadata(PATH + dataset + '/metadata.json')
        resolution = metadata['resolution']
        frame_cnt = metadata['frame count']
        frame_rate = metadata['frame rate']
        original_resol = str(resolution[1]) + 'p'
        gt_file = PATH  + dataset + '/' + original_resol + '/profile/updated_gt_FasterRCNN_COCO.csv'
        gt, nb_frames = load_full_model_detection_new(gt_file)
        #nb_frames = 1800
        print(len(gt), nb_frames)
        # pdb.set_trace()
        start_frame = 1 + OFFSET * frame_rate
        #end_frame = start_frame + TEST_LENGTH * frame_rate - 1
        end_frame = start_frame + 240 - 1
        original_bw = compute_video_size(PATH+dataset+'/'+original_resol +'/',
                                         start_frame, end_frame, frame_rate,
                                         frame_rate, resolution)

        bw_540 = compute_video_size(PATH+dataset+'/'+RESOLUTION_LIST[1]+'/',
                                    start_frame, end_frame, frame_rate,
                                    frame_rate,
                                    RESOLUTION_DICT[RESOLUTION_LIST[1]])
        #bw_480 = compute_video_size(PATH+dataset+'/'+RESOLUTION_LIST[1]+'/',
        #                            901, nb_frames, frame_rate, frame_rate, RESOLUTION_DICT[RESOLUTION_LIST[1]])
        #pdb.set_trace()

        bw = compute_video_size(PATH+dataset+'/'+RESOLUTION_LIST[0]+'/',
                                start_frame, end_frame, frame_rate, frame_rate,
                                RESOLUTION_DICT[RESOLUTION_LIST[0]])
        print(original_bw, bw_540/original_bw,  bw/original_bw)

        with open('SimpleProto_result_{}.csv'.format(dataset), 'w') as f:
            f.write('alpha, beta, tp,fp,fn,precision,recall,f1,bw\n')
            for beta in [0.5]: # [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                tp = {}
                fp = {}
                fn = {}
                resol_counts = defaultdict(int)
                resol_to_cropped_imgs = defaultdict(list)

                os.system('rm crop_images/*/*.jpg')

                for frame_idx in range(start_frame, end_frame + 1):
                    gt_boxes = gt[frame_idx]
                    # tmp = []
                    # for i, box in enumerate(gt_boxes):
                    #     if box[5] >= BETA:
                    #         tmp.append(gt_boxes[i])
                    # gt_boxes = tmp
                    detected_boxes = list()
                    cropped_regions = list() # record the cropped regions that
                    # need to be checked in higher resolution
                    for resol_idx, resol in enumerate(RESOLUTION_LIST):
                        #send the frame to server at low resolution
                        dt_file = PATH + dataset + '/' + resol + '/profile/updated_gt_FasterRCNN_COCO.csv'
                        #print(dt_file)
                        dt, _ = load_full_model_detection_new(dt_file)
                        boxes = dt[frame_idx]
                        debug_print('{} {}'.format(resol, boxes))

                        if resol_idx == 0:
                            # lowest resolution and entire frame is transmitted
                            #bw += RESOLUTION_DICT[resol][0]*RESOLUTION_DICT[resol][1]
                            for box in boxes:
                                score = box[5]
                                if score >= beta:
                                    # no need to further send cropped area at higher resolution
                                    box[:4] = scale(box[:4], RESOLUTION_DICT[resol], resolution)
                                    detected_boxes.append(box)
                                    resol_counts[resol] += 1
                                elif score >= ALPHA and score < beta:
                                    # TODO: need to keep a small margin and the boundary need
                                    cropped_regions.append(box)
                        else:

                            prev_resol = RESOLUTION_DICT[RESOLUTION_LIST[resol_idx-1]]
                            cur_resol = RESOLUTION_DICT[resol]
                            cropped_regions_left = []
                            for region in cropped_regions:
                                # pdb.set_trace()
                                region[:4] = scale(region[:4], prev_resol, cur_resol)
                                if resol != 'original':
                                    img_in = PATH + dataset + '/' + resol +'/{:06d}.jpg'.format(frame_idx)
                                else:
                                    img_in = PATH + dataset + '/{:06d}.jpg'.format(frame_idx)
                                img_out = 'crop_images/{}/{:06d}.jpg'.format(resol, len(resol_to_cropped_imgs[resol]) + 1)
                                print(img_out)
                                resol_to_cropped_imgs[resol].append(img_out)
                                crop_image(img_in, region[0], region[1],
                                           region[2], region[3], cur_resol[0],
                                           cur_resol[1], img_out)
                                overlaps = []
                                for box in boxes:
                                    overlaps.append(overlap_percentage(box[:4], region[:4]))
                                debug_print('{} {}: {}'.format(resol, 'overlap', overlaps))

                                if overlaps:
                                    max_overlap_idx = np.argmax(overlaps)
                                    debug_print('max overlap index={}'.format(max_overlap_idx))
                                    box = boxes[max_overlap_idx]
                                    score = box[5]
                                    if overlaps[max_overlap_idx] >= 0.3: # score >= beta and
                                        box[:4] = scale(box[:4], RESOLUTION_DICT[resol], resolution)
                                        detected_boxes.append(box)
                                        del boxes[max_overlap_idx]
                                        resol_counts[resol] += 1
                                    elif score >= ALPHA and score < beta and overlaps[max_overlap_idx] != 0.0:
                                        # xmax = box[0] + box[2]
                                        # ymax = box[1] + box[3]
                                        # box[0] = min(0, box[0]-10)
                                        # box[1] = min(0, box[1]-10)
                                        # xmax = xmax + 10
                                        # ymax = ymax + 10
                                        # box[2] = max(RESOLUTION_DICT[resol][0], xmax) - box[0]
                                        # box[3] = max(RESOLUTION_DICT[resol][1], ymax) - box[1]
                                        #cropped_regions_left.append(box)
                                        pass
                                    else:
                                        pass # just drop the cropped area
                                else:
                                    # Server didnt even find an object in the cropped
                                    # area by using higher resolution
                                    # So just drop the area
                                    pass

                            cropped_regions = cropped_regions_left

                        debug_print('{} frame_id={} detected: {}'.format(resol, frame_idx, detected_boxes))
                        debug_print('{} frame_id={} cropped: {}'.format(resol, frame_idx, cropped_regions))
                        debug_print('{} frame_id={} gt: {}'.format('original', frame_idx, gt_boxes))
                        if not cropped_regions:
                            break
                    #pdb.set_trace()
                    tp[frame_idx], fp[frame_idx], fn[frame_idx] = eval_single_image(gt_boxes, detected_boxes)
                    debug_print('tp={}, fp={}, fn={}'.format(tp[frame_idx], fp[frame_idx], fn[frame_idx]))
                # TODO: Compress croped images at different resolutions
                # pdb.set_trace()
                for resol in RESOLUTION_LIST[1:]:
                    if resol_to_cropped_imgs[resol]:
                        bw += compress_images_to_video_v1('crop_images/'+resol+'/', 1, len(resol_to_cropped_imgs[resol]),
                        frame_rate, str(RESOLUTION_DICT[resol][0])+'x'+str(RESOLUTION_DICT[resol][1]), 25, 'tmp.mp4')

                tp_total = sum(tp.values())
                fp_total = sum(fp.values())
                fn_total = sum(fn.values())

                f1 = compute_f1(tp_total, fp_total, fn_total)
                recall = tp_total / (tp_total + fn_total)
                precison = tp_total / (tp_total + fp_total)
                print('tp={}, fp={}, fn={}, f1={}'.format(tp_total, fp_total, fn_total, f1))
                print('relative bandwidth={}'.format(bw/original_bw))
                f.write(','.join([str(ALPHA), str(beta), str(tp_total),
                                  str(fp_total), str(fn_total), str(precison),
                                  str(recall), str(f1), str(bw/original_bw)+'\n']))
                print(resol_counts)

if __name__ == "__main__":
    main()
