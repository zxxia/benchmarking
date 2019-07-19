# Benchmarking

Two parts of the code:

### 1. Video download, label and profile.

To build a dataset from a youtube video:
1. use **youtube-dl** to download the video.
  
   1. Install youtube-dl
   
   2. Use ``` youtube-dl -F <youtube URL> ``` to list all formats, and choose a format id (e.g. 96).
   
   3.  If it is a live video, use the following command
      ```ffmpeg -i $(youtube-dl -f {format_id} -g <youtube URL>) -c copy -t 00:20:00 output.ts```
   
   4. if not live video, run 
   
      ```ffmpeg -i $(youtube-dl -f {format_id} -g <youtube URL>) -c copy {VIDEONAME}.mp4```
   
      
   
2. use ffmpeg to extract the frames
   ```ffmpeg -i <Video Filename> %06d.jpg -hide_banner```

3. Use metadata_generator.py in new_video folder to generate metadta for the video. Metadata should be generated in the same folder of the input video.
   ```python3 metadata_generator.py [Path to input video] [Output path of output json metadata file]```

4. Config new_video/generate_ground_truth.sh to run Faster-RCNN model on extracted frames. 

   1. Get this repo https://github.com/tensorflow/models/tree/master/research/object_detection 
   2. Download this trained model http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz, and put it under *models/research/object_detection*, then uncompress the model file.
   3. Change CODE_PATH, FULL_MODEL_PATH, DATA_PATH, DATASET_LIST to your own path.

5. Objects in a typical traffic video (and corresponding label index in COCO)

   1 person

   2 bicycle

   3 car

   4 motorcycle

   6 bus

   8 truck





### 2. Individual pipelines

1. VideoStorm (Temporal sampling)
   1. Use videostorm/VideoStorm_temporal.py 
   2. VideoStorm reduces video frame rate to reduce cost. However, reducing frame rate could possiblely make the accuracy drop. Therefore, we can get a cost-accuracy curve by varing the sampling rate (=frame rate after sampling/original frame rate). 
   3. **How to use code:** 
      1. Change *dataset_list*, *data_path* to your video name and path.
      2. If plot F1_score_list and frame_rate_list, you can see the cost-accuracy curve.
      3. We assume there is a requirement for accuracy (F1 score). The target f1 score is 0.9, then we compute the **minimum frame rate** needed to achieve f1=0.9.
      4. An example result file should be in same format as VideoStorm_result_tmp.csv.
2. Glimpse
   1. Use glimpse_youtube.py.
   2. Key idea of Glimpse is to send selected frames to server for detection, and run tracking on unselected frames. Therefore, there are two important parts of glimpse, **compute frame difference** and **tracking**. We could get the cost-accuracy curve by varing two parameters.
      1. frame difference threshold.
      2. Tracking error threshold.
   3. **How to use code:**
      1. Change  *video_type*, *path* to your video name and path.
      2. If plot f1_list and frame_rate_list, you can see the cost-accuracy curve.
3. NoScope
4. Fast cascading
5. AWStream
   1. Key idea: reduces video frame rate and frame resolution to reduce bandwidth. However, reducing frame rate and frame resolution could possiblely make the accuracy drop. Therefore, we can get a cost (bandwidth)-accuracy curve by varing the sampling rate and sampling rate.
   2. Before use AWStream code, we need to run full model on different resolution videos. **Run generate_ground_truth.sh** again. I changed the code. Set —resize to True and add RESIZE_RESOL to specify the resulting resolution. 
   3. (Important!!) I used **resize.py** to resize the original video.  However, the image size becomes larger after resizing. The reason is that cv2.imwrite will write the resized image at quality level 95 (possibly higher than original image). So please check how to use **ffmpeg** to resize images. Check */home/zhujun/video_analytics_pipelines/dataset/Youtube/highway/540p* for example format.
   4. **Run awstream.py**. Update image_resolution_dict and dataset_list to add your own video. Also, update *path*.
   5. This code is not finalized. Try this code with a small video. 
6. HotCloud
