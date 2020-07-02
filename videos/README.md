# Videos and Dataset Abstraction

## Base Class
```video.py``` is the base class of all dataset wrapper. 

## KITTI
```kitti.py``` is the wrapper of KITTI Dataset. 

## MOT15
```mot15.py``` is the wrapper of MOT15 Dataset. 

## MOT16
```mot16.py``` is the wrapper of MOT16 Dataset. 

## Waymo Open Dataset
```waymo.py``` is the wrapper of a Waymo Open Dataset. 

## Youtube
```youtube.py``` is the wrapper of a youtube downloaded video. 

### Directory Tree

``` text
tv_show
|   tv_show.mp4
|   tv_show_1280x720_23.mp4
|   tv_show_960x540_23.mp4
|   ...
|
----profile
|   | faster_rcnn_resenet101_1280x720_23_detections.csv
|   | faster_rcnn_resenet101_1280x720_23_smoothed_detections.csv
|   | faster_rcnn_resenet101_1280x720_23_profile.csv
|   | ...
|
|---720p
|   | 1280x720 frame images ...
|   |
|---540p
|   | 960x540 frame images ...
```
