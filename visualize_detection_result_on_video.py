#visualize_detection_result_on_video.
#input: detection csv, mp4 video
#output: each frame labled
import os
import glob
import cv2
import csv
# import seaborn as sns
import copy
from scipy import stats
import matplotlib.pyplot as plt
from object_detection.infer import load_object_detection_results 
detection_filepath="/Users/mac/Downloads/大二下/UChicago/benchmarking/data/Videos/motorway/profile/ssd_mobilenet_v2_1280x720_23_smoothed_detections.csv" #ssd_mobilenet_v2_1280x720_23_smoothed_detections.csv"#faster_rcnn_resnet101_1280x720_23_smoothed_detections.csv
video_filepath="/Users/mac/Downloads/大二下/UChicago/benchmarking/data/Videos/motorway/motorway.mp4"
output_video_filepath="/Users/mac/Downloads/大二下/UChicago/benchmarking/data/Videos/motorway/motorway_ssd.mp4"
detection_dict=load_object_detection_results(detection_filepath)
cap = cv2.VideoCapture(video_filepath)
frameID=0
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_video_filepath,fourcc, 20.0, (1280,720))
while(cap.isOpened()):
    ret, frame = cap.read()
    frameID+=1
    # print(detection_dict[frameID])
    for xmin, ymin, xmax, ymax, t, score, obj_id in detection_dict[frameID]:
        # xmin, ymin, xmax, ymax, t, score, obj_id = detection_dict[frameID]
        # print("xmin, ymin, xmax, ymax, t, score, obj_id", xmin, ymin, xmax, ymax, t, score, obj_id)
        xmin=int(xmin)
        ymin=int(ymin)
        xmax=int(xmax)
        ymax=int(ymax)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0))
        #font = cv2.FONT_HERSHEY_SUPLEX
        text="t:%lf score: %f obj_id: %f" % (t, score, obj_id)
        cv2.putText(frame, text , (xmin, ymin), 1,1, (0,0,255), 1)
    
    out.write(frame)    
    # cv2.imshow('frame',frame)
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break
    
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()