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
   ffmpeg -i <Video Filename> %06d.jpg -hide_banner

3. Config new_video/generate_ground_truth.sh to run Faster-RCNN model on extracted frames. 

   1. Get this repo https://github.com/tensorflow/models/tree/master/research/object_detection 
   2. Download this trained model http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz, and put it under *models/research/object_detection*, then uncompress the model file.
   3. Change CODE_PATH, FULL_MODEL_PATH, DATA_PATH, DATASET_LIST to your own path.





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
3. NoScope
4. Fast cascading
5. AWStream
6. HotCloud