import cv2
import time
import numpy as np
from collections import defaultdict
import os
from my_utils import interpolation, IoU
from matplotlib import pyplot as plt

# compute frame difference
def frame_difference(old_frame, new_frame, thresh=35):
  # thresh = 35 is used in Glimpse paper
  diff = np.absolute(new_frame - old_frame)
  return np.sum(np.greater(diff, thresh))


# compute target frame rate when target f1 is achieved
def compute_target_frame_rate(frame_rate_list, f1_list, target_f1=0.9):
  index = frame_rate_list.index(max(frame_rate_list))
  f1_list_normalized = [x/f1_list[index] for x in f1_list]
  result = [(y,x) for x, y in sorted(zip(f1_list_normalized, 
       frame_rate_list))]
  # print(list(zip(frame_rate_list,f1_list_normalized)))
  frame_rate_list_sorted = [x for (x,_) in result]
  f1_list_sorted = [y for (_,y) in result]
  index = next(x[0] for x in enumerate(f1_list_sorted) if x[1] > target_f1)
  if index == 0:
    target_frame_rate = frame_rate_list_sorted[0]
  else:
    point_a = (f1_list_sorted[index-1], frame_rate_list_sorted[index-1])
    point_b = (f1_list_sorted[index], frame_rate_list_sorted[index])

    target_frame_rate = interpolation(point_a, point_b, target_f1)
  return target_frame_rate


# write glimpse detection results to file
def write_pipeline_result(frame_start, frame_end, 
  gt_annot, dt_glimpse, frame_flag, csvf):
  for i in range(frame_start, frame_end + 1):
    csvf.write(str(i) + ',')
    gt_boxes_final = gt_annot[i].copy()
    dt_boxes_final = dt_glimpse[i].copy()


    gt_string = []
    for box in gt_boxes_final:
      gt_string.append(' '.join([str(x) for x in box]))

    dt_string = []
    for box in dt_boxes_final:
      dt_string.append(' '.join([str(x) for x in box]))


    csvf.write(';'.join(gt_string) + ',')
    csvf.write(';'.join(dt_string) + ',' + str(frame_flag[i]) + '\n')
  csvf.close()
  return


def findDistance(r1,c1,r2,c2):
  d = (r1-r2)**2 + (c1-c2)**2
  d = d**0.5
  return d


# for each box, track it
def tracking(oldFrameGray, newFrameGray, old_corners, tracking_error_thresh, 
  image_resolution, h):
  lk_params = dict(maxLevel = 2)
  new_corners, st, err = cv2.calcOpticalFlowPyrLK(oldFrameGray, 
                          newFrameGray, 
                          old_corners, 
                          None, 
                          **lk_params)
  new_corners = new_corners[st==1].reshape(-1,1,2)
  old_corners = old_corners[st==1].reshape(-1,1,2)
  if len(new_corners) < 4:
    # print('No enough feature points')
    return [0,0,0,0] # this object disappears

  r_add,c_add = 0,0
  for corner in new_corners:
    r_add = r_add + corner[0][1]
    c_add = c_add + corner[0][0]
  centroid_row = int(1.0*r_add/len(new_corners))
  centroid_col = int(1.0*c_add/len(new_corners))
  #draw centroid
  #cv2.circle(img,(int(centroid_col),int(centroid_row)),5,(255,0,0)) 
  #add only those corners to new_corners_updated 
  #which are at a distance of 30 or lesse

  new_corners_updated = new_corners.copy()
  old_corners_updated = old_corners.copy()

  tobedel = []
  dist_list = []

  for index in range(len(new_corners)):
    # remove coners that are outside the image
    if new_corners[index][0][0] > image_resolution[0] or \
       new_corners[index][0][0] < 0 or \
       new_corners[index][0][1] > image_resolution[1] or \
       new_corners[index][0][1] < 0:
      tobedel.append(index)

    # remove outliers
    if findDistance(new_corners[index][0][1],
            new_corners[index][0][0],
            int(centroid_row),
            int(centroid_col)) > 2*h:
      tobedel.append(index)



    dist = np.linalg.norm(new_corners[index] 
                - old_corners[index])
    dist_list.append(dist)

  dist_median = np.median(dist_list)
  dist_std = np.std(dist_list)

  for index in range(len(new_corners)):
    if dist_list[index] > dist_median + 3*dist_std or \
      dist_list[index] < dist_median - 3*dist_std:
      tobedel.append(index)


  new_corners_updated = np.delete(new_corners_updated,tobedel,0)
  old_corners_updated = np.delete(old_corners_updated,tobedel,0)

  # if there are not enough feature points, then assume this object disappears
  # in current frame
  if len(new_corners_updated) < 4:
    # print('No enough feature points')
    return [0,0,1,1] # this object disappears

  x_list = []
  y_list = []
  for corner in new_corners_updated:
    # cv2.circle(newFrameGray, (int(corner[0][0]),int(corner[0][1])) ,5,(0,255,0))
    x_list.append(int(corner[0][0]))
    y_list.append(int(corner[0][1]))

  dist_list = []
  for index in range(len(new_corners_updated)):
    dist = np.linalg.norm(new_corners_updated[index] 
                - old_corners_updated[index])
    dist_list.append(dist)
  # print([(x,y) for (x,y) in zip(new_corners_updated,old_corners_updated)])
  # print(np.std(dist_list))
  if np.std(dist_list) > tracking_error_thresh: # tracking failure, this is a trigger frame
    return [0,0,0,0] # indicates tracking failure 
  else:
    x = min(x_list)
    y = min(y_list)
    w = max(x_list) - x
    h = max(y_list) - y
    return [x, y, w, h]
  # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)


def eval_single_image_single_type(gt_boxes, pred_boxes, iou_thresh):
  gt_idx_thr = []
  pred_idx_thr = []
  ious = []
  for ipb, pred_box in enumerate(pred_boxes):
    for igb, gt_box in enumerate(gt_boxes):
      box1 = pred_box.copy()
      box1[2] += box1[0]
      box1[3] += box1[1]  
      box2 = gt_box.copy()
      box2[2] += box2[0]
      box2[3] += box2[1]    
      iou = IoU(box1, box2)
      if iou > iou_thresh:
        gt_idx_thr.append(igb)
        pred_idx_thr.append(ipb)
        ious.append(iou)

  args_desc = np.argsort(ious)[::-1]
  if len(args_desc) == 0:
    # No matches
    tp = 0
    fp = len(pred_boxes)
    fn = len(gt_boxes)
  else:
    gt_match_idx = []
    pred_match_idx = []
    for idx in args_desc:
      gt_idx = gt_idx_thr[idx]
      pr_idx = pred_idx_thr[idx]
      # If the boxes are unmatched, add them to matches
      if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
        gt_match_idx.append(gt_idx)
        pred_match_idx.append(pr_idx)
    tp = len(gt_match_idx)
    fp = len(pred_boxes) - len(pred_match_idx)
    fn = len(gt_boxes) - len(gt_match_idx)
  return tp, fp, fn

def eval_single_image(gt_boxes, dt_boxes, iou_thresh=0.5):
  tp_dict = {}
  fp_dict = {}
  fn_dict = {}
  gt = defaultdict(list)  
  dt = defaultdict(list)
  for box in gt_boxes:
    gt[box[4]].append(box[0:4])
  for box in dt_boxes:
    dt[box[4]].append(box[0:4])

  for t in gt.keys():
    current_gt = gt[t]
    current_dt = dt[t]
    tp_dict[t], fp_dict[t], fn_dict[t] = eval_single_image_single_type(
                       current_gt, current_dt, iou_thresh)

  tp = sum(tp_dict.values())
  fp = sum(fp_dict.values())
  fn = sum(fn_dict.values())
  extra_t = [t for t in dt.keys() if t not in gt]
  for t in extra_t:
    fp += len(dt[t])

  return tp, fp, fn









def eval_pipeline_accuracy(frame_start, frame_end, gt_annot, dt_glimpse, iou_thresh=0.5):
  tp = defaultdict(int)
  fp = defaultdict(int)
  fn = defaultdict(int)
  gt_cn = 0
  dt_cn = 0
  for i in range(frame_start, frame_end + 1):

    gt_boxes_final = gt_annot[i].copy()
    dt_boxes_final = dt_glimpse[i].copy()
    gt_cn += len(gt_boxes_final)
    dt_cn += len(dt_boxes_final)


    tp[i], fp[i], fn[i] = eval_single_image(gt_boxes_final, dt_boxes_final, iou_thresh)

  tp_total = sum(tp.values())
  fn_total = sum(fn.values())
  fp_total = sum(fp.values())
  if gt_cn == 0:
    if dt_cn == 0:
      return 1
    else:
      return 0      


  if tp_total == 0:
    f1 = 0
  else:
    precison = float(tp_total) / (tp_total + fp_total)
    recall = float(tp_total) / (tp_total + fn_total)
    f1 = 2*(precison*recall)/(precison+recall) 

  return f1

def pipeline(img_path, dt_annot, gt_annot, frame_start, frame_end, csvf, 
  image_resolution, frame_rate, frame_difference_thresh, 
  tracking_error_thresh, kitti=True):

  frame_flag = {}
  dt_glimpse = defaultdict(list)
  triggered_frame = 0
  cn = 0

  # start from the first frame
  # The first frame has to be sent to server. 
  if kitti:
    img_name = format(frame_start, '010d') + '.png'
  else:
    img_name = format(frame_start, '06d') + '.jpg'

  oldFrameGray = cv2.imread(img_path + img_name, 0)
  lastTriggeredFrameGray = oldFrameGray.copy()
  dt_glimpse[frame_start] = dt_annot[frame_start] # get detection from server
  frame_flag[frame_start] = 1    # if detection is obtained from server, set frame_flag to 1
  triggered_frame += 1 # count triggered fames
  cn += 1 
  last_index = frame_start
  # run the pipeline for the rest of the frames
  for i in range(frame_start + 1, frame_end + 1):

    if kitti:
      newImgName = img_path + format(i, '010d') + '.png'
    else:
      newImgName = img_path + format(i, '06d') + '.jpg'
    newFrameGray = cv2.imread(newImgName, 0)

    # compute frame difference
    frame_diff = frame_difference(lastTriggeredFrameGray, 
                    newFrameGray)

    if frame_diff > frame_difference_thresh:
      # triggered
      # run inference to get the detection results
      dt_glimpse[i] = dt_annot[i].copy() 
      triggered_frame += 1
      oldFrameGray = newFrameGray.copy()
      lastTriggeredFrameGray = oldFrameGray.copy()
      frame_flag[i] =  1
    else: 
      # use tracking to get the result
      assert last_index in dt_glimpse, print(last_index)
      if not dt_glimpse[last_index] : 
      # last frame is empty, then current frame is empty
        dt_glimpse[i] = []
        oldFrameGray = newFrameGray.copy()
        frame_flag[i] =  0
      else: 
        # need to use tracking to get detection results
        # track from last frame
        for [x, y, w, h, t] in dt_glimpse[last_index]: #dt_annot[i - 1]:            
          roi = oldFrameGray[y:y+h,x:x+w]
          old_corners = cv2.goodFeaturesToTrack(roi, 26, 0.01, 7) #find corners

          # add four corners of the bounding box as the feature points
          if old_corners is not None:
            old_corners[:,0,0] = old_corners[:,0,0] + x
            old_corners[:,0,1] = old_corners[:,0,1] + y
            old_corners = np.append(old_corners, [[[np.float32(x), 
                        np.float32(y)]]], axis=0)

          else:
            old_corners = [[[np.float32(x) , np.float32(y)]]]
          old_corners = np.append(old_corners, [[[np.float32(x+w), 
                      np.float32(y)]]], axis=0)
          old_corners = np.append(old_corners, [[[np.float32(x), 
                      np.float32(y+h)]]], axis=0)
          old_corners = np.append(old_corners, [[[np.float32(x+w), 
                      np.float32(y+h)]]], axis=0)
          # print(len(old_corners), old_corners[0])

          # No need to read tracking code
          [newX, newY, newW, newH] = tracking(oldFrameGray, 
                            newFrameGray, 
                            old_corners, 
                            tracking_error_thresh, 
                            image_resolution, 
                            h)
          if not (newX+newY+newW+newH): # track failure
            track_success_flag = 0
            break
          elif newW < 15 or newH < 15: # object too small = it disappears
            track_success_flag = 1
          else:
            # add tracking result as glimpse result
            dt_glimpse[i].append([newX, newY, newW, newH, t])
            track_success_flag = 1
        if track_success_flag == 0:
          # tracking fails, get detection result from server
          dt_glimpse[i] = dt_annot[i].copy()
          triggered_frame += 1
          frame_flag[i] =  1
          lastTriggeredFrameGray = oldFrameGray.copy()
        else:
          if i not in dt_glimpse:
            dt_glimpse[i] = []
          frame_flag[i] =  0
        oldFrameGray = newFrameGray.copy()
    last_index = i
    # print(i, frame_flag[i], len(dt_glimpse[i]))
    # # if [] in dt_glimpse[i]:
    # #   continue
    # for [x,y,w,h,t] in dt_glimpse[i]:
    #   cv2.rectangle(newFrameGray, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # # if [] not in gt_annot[i]:
    # #   for [x,y,w,h] in gt_annot[i]:
    # #     cv2.rectangle(newFrameGray, (x, y), (x + w, y + h), (255, 255, 0), 3)
    # cv2.imshow(str(i) + '_' + str(frame_flag[i]), newFrameGray) 
    # cv2.waitKey()

    # total_time += (time.time()-start_time)*1000
  print("triggered frames", triggered_frame)

  write_pipeline_result(frame_start, 
              frame_end, 
              gt_annot, 
              dt_glimpse, 
              frame_flag, 
              csvf)
  f1 = eval_pipeline_accuracy(frame_start,
                frame_end, 
                gt_annot, 
                dt_glimpse)


  return triggered_frame, f1



def get_gt_dt(annot_path):
  # read ground truth and full model (server side) detection results
  gt_annot = defaultdict(list)
  dt_annot = defaultdict(list)
  frame_end = 0
  with open(annot_path, 'r') as f:
    for line in f:
      annot_list = line.strip().split(',')
      frame_id = int(annot_list[0].replace('.png',''))
      frame_end = frame_id
      dt_str = annot_list[1]
      gt_str = annot_list[2]
      gt_boxes = gt_str.split(';')
      if gt_boxes == ['']:
        gt_annot[frame_id] = []
      else:
        for box in gt_boxes:
          box_list = box.split(' ')
          x = int(box_list[0])
          y = int(box_list[1])
          w = int(box_list[2])
          h = int(box_list[3])
          t = int(box_list[4])
          if t == 1:
            gt_annot[frame_id].append([x,y,w,h])


      dt_boxes = dt_str.split(';')
      if dt_boxes == ['']:
        dt_annot[frame_id] = []
      else:
        for box in dt_boxes:
          box_list = box.split(' ')
          x = int(box_list[0])
          y = int(box_list[1])
          w = int(box_list[2])
          h = int(box_list[3])
          t = int(box_list[4])
          if t == 1:
            dt_annot[frame_id].append([x,y,w,h])
  return gt_annot, dt_annot, frame_end

def read_object_id(annot_path):
  object_to_frame = defaultdict(list)
  with open(annot_path, 'r') as f:
    f.readline() 
    for line in f:
      line_list = line.split(',')
      frame_id = int(line_list[0])
      object_id = int(line_list[1])
      object_to_frame[object_id].append(frame_id)
  return object_to_frame

def main():
  image_resolution = [1242, 375]
  video_index_dict = {'Road': [15,27,28,29,32,52],
          'City': [1,2,5,9,11,13,14,17,18,48,51,56,57,59,60,84,91,93],
          'Residential': [19,20,22,23,35,36,39,46,61,64,79,86,87]}
  # path =  '/home/zhujun/video_analytics_pipelines/dataset/KITTI/'
  path = '/Users/zhujunxiao/Desktop/benchmarking/KITTI/'
  inference_time = 100 # avg. GPU processing  time
  final_result_f = open('KITTI_glimpse_result.csv','w')
  frame_rate = 10
  target_f1 = 0.9

  for video_type in ['City', 'Residential','Road']:
    for i in video_index_dict[video_type]:
      video_index = format(i,'04d') 
      print(video_index)
      # read ground truth and full model detection result
      # image name, detection result, ground truth
      annot_path =path + video_type +'/2011_09_26_drive_' \
                + video_index + '_sync/result/input_w_gt.csv'
      img_path = path + video_type +'/2011_09_26_drive_' \
                + video_index + '_sync/image_02/data/'
      gt_annot, dt_annot, frame_end = get_gt_dt(annot_path)


      # Run inference on the first frame
      # Two parameters: frame difference threshold, tracking error thresh
      para1_list = [1,1.2,1.3,1.4,1.5,2,2.2,2.3,2.5,4,4.5,5]
      para2_list = [0.5,1,5,10]

      result_file = './result/' + video_type + '_' + video_index \
              + '_glimpse.csv' 
      result_f = open(result_file, 'w')
      result_f.write('frame difference factor,'\
               'tracking error thresh, avg Frame rate, f1 score\n')
      result_direct = annot_path.replace('result/input_w_gt.csv',
                         'glimpse_result_0609/')
      if not os.path.exists(result_direct):
        os.makedirs(result_direct)

      frame_rate_list = []
      f1_list = []
      for para1 in para1_list:
        for para2 in para2_list:
          result_f.write(str(para1) + ',' + str(para2) + ',')
          dt_glimpse_file = result_direct + str(para1) + '_'+ \
                    str(para2) + '.csv'
          csvf = open(dt_glimpse_file, 'w')
          print(para1, para2)
          # larger para1, smaller thresh, easier to be triggered
          frame_difference_thresh = \
                image_resolution[0]*image_resolution[1]/para1 
          tracking_error_thresh = para2


          triggered_frame, f1 = pipeline(img_path,
                            dt_annot, 
                            gt_annot,
                            0,
                            frame_end, 
                            csvf, 
                            image_resolution,
                            frame_rate,
                            frame_difference_thresh, 
                            tracking_error_thresh)

          current_frame_rate = triggered_frame/(frame_end + 1) * frame_rate
          print("frame rate, f1 score:", current_frame_rate, f1)
          frame_rate_list.append(current_frame_rate)
          f1_list.append(f1)
          result_f.write(str(current_frame_rate)+','+str(f1)+'\n')  
      result_f.close()

      target_frame_rate = compute_target_frame_rate(frame_rate_list,
                              f1_list,
                              target_f1)
      plt.scatter(f1_list, frame_rate_list)
      plt.show()
      final_result_f.write(video_type+'_'+str(video_index)+','+str(target_frame_rate)+'\n')
      print(target_frame_rate)


  return











if __name__=='__main__':
  main()