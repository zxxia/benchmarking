import argparse
import cv2
import glob
import logging
import json
import os
import time
from benchmarking.constants import babygroot_DT_ROOT
from benchmarking.ground_truth.run_inference import run_inference
from benchmarking.vigil.run_Vigil import run_Vigil
from benchmarking.glimpse.run_Glimpse import run_Glimpse


def read_video_info(video_path):
    video_info = {}
    vid = cv2.VideoCapture(video_path)
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vid.get(cv2.CAP_PROP_FPS))      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    vid.release()
    video_info['video_name'] = os.path.basename(video_path).replace('.mp4', '')
    video_info['duration'] = duration
    video_info['resol'] = [width, height]
    video_info['frame_rate'] = fps
    video_info['frame_count'] = frame_count
    return video_info

def read_image_info(img_files, frame_rate):
    image_info = {}
    image_info['frame_count'] = len(img_files)
    image_info['frame_rate'] = frame_rate

    img = cv2.imread(img_files[0])
    height, width, channels = img.shape

    duration = image_info['frame_count']/frame_rate
    image_info['duration'] = duration
    image_info['resol'] = [width, height]
    return image_info

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="VideoStorm with temporal overfitting")
    parser.add_argument("--gpu_num", type=str, required=True, default="3")
    parser.add_argument("--name", type=str, required=True, help="dataset name")
    parser.add_argument("--type", type=str, required=True, default="video")
    parser.add_argument("--camera", type=str, required=True, default=None,
                        help="camera type")
    parser.add_argument("--frame_rate", type=int, default=0,
                        help="frame rate")
    args = parser.parse_args()
    return args

def preprocessing(path, _type, frame_rate, camera):
    if _type == 'video':
        mp4_files = glob.glob(path + '/*.mp4')
        assert len(mp4_files) == 1, logging.ERROR('No mp4 found.')
        dataset_info = read_video_info(mp4_files[0])

    else:
        img_files = glob.glob(path + '/*.' + _type)
        assert len(img_files) >= 1, logging.ERROR('No images found.')
        assert frame_rate != 0, print(frame_rate)
        dataset_info = read_image_info(img_files, frame_rate)

    dataset_info['type'] = _type
    dataset_info['camera_type'] = camera
    dataset_info['path'] = path
    return dataset_info



def main():
    args = parse_args()
    # define video path, name
    # video source format

    waymo_path = '/mnt/data/zhujun/dataset/Waymo/waymo_images'
    folder_list = sorted([x for x in os.listdir(waymo_path) if os.path.isdir(os.path.join(waymo_path, x))])

    selected = ['segment-10206293520369375008_2796_800_2816_800_with_camera_labels', 'segment-11004685739714500220_2300_000_2320_000_with_camera_labels', 'segment-12681651284932598380_3585_280_3605_280', 'segment-12974838039736660070_4586_990_4606_990', 'segment-13177337129001451839_9160_000_9180_000', 'segment-13258835835415292197_965_000_985_000', 'segment-14098605172844003779_5084_630_5104_630', 'segment-14466332043440571514_6530_560_6550_560', 'segment-14742731916935095621_1325_000_1345_000', 'segment-15367782110311024266_2103_310_2123_310', 'segment-2590213596097851051_460_000_480_000', 'segment-2692887320656885771_2480_000_2500_000', 'segment-3418007171190630157_3585_530_3605_530', 'segment-3437741670889149170_1411_550_1431_550', 'segment-6814918034011049245_134_170_154_170', 'segment-759208896257112298_184_000_204_000', 'segment-809159138284604331_3355_840_3375_840', 'segment-17135518413411879545_1480_000_1500_000', 'segment-12866817684252793621_480_000_500_000', 'segment-10723911392655396041_860_000_880_000', 'segment-6038200663843287458_283_000_303_000', 'segment-6128311556082453976_2520_000_2540_000', 'segment-7517545172000568481_2325_000_2345_000', 'segment-7861168750216313148_1305_290_1325_290', 'segment-4167304237516228486_5720_000_5740_000', 'segment-9415086857375798767_4760_000_4780_000', 'segment-3451017128488170637_5280_000_5300_000', 'segment-16153607877566142572_2262_000_2282_000', 'segment-16105359875195888139_4420_000_4440_000', 'segment-2681180680221317256_1144_000_1164_000', 'segment-5990032395956045002_6600_000_6620_000', 'segment-5302885587058866068_320_000_340_000', 'segment-4759225533437988401_800_000_820_000', 'segment-15533468984793020049_800_000_820_000_with_camera_labels', 
        'segment-14233522945839943589_100_000_120_000_with_camera_labels', 'segment-15578655130939579324_620_000_640_000_with_camera_labels', 'segment-13476374534576730229_240_000_260_000', 'segment-17244566492658384963_2540_000_2560_000', 
        'segment-4487677815262010875_4940_000_4960_000', 'segment-4916527289027259239_5180_000_5200_000', 'segment-4641822195449131669_380_000_400_000', 'segment-8327447186504415549_5200_000_5220_000', 'segment-8207498713503609786_3005_450_3025_450', 'segment-5222336716599194110_8940_000_8960_000', 'segment-4986495627634617319_2980_000_3000_000', 'segment-2064489349728221803_3060_000_3080_000_with_camera_labels', 'segment-3908622028474148527_3480_000_3500_000_with_camera_labels', 'segment-2922309829144504838_1840_000_1860_000', 'segment-15834329472172048691_2956_760_2976_760', 'segment-17386718718413812426_1763_140_1783_140', 'segment-17066133495361694802_1220_000_1240_000', 'segment-12161824480686739258_1813_380_1833_380', 'segment-11918003324473417938_1400_000_1420_000', 'segment-11037651371539287009_77_670_97_670_with_camera_labels', 'segment-16213317953898915772_1597_170_1617_170_with_camera_labels']
    not_selected = [x for x in folder_list if x not in selected]
    start_time = time.time()
    profile_length = 10
    segment_length = 30
    target_f1 = 0.9
    for folder in selected[0:2]: #folder_list[0:1]:



        path = os.path.join(waymo_path, folder, 'FRONT')
        print(path)
        logging.basicConfig(filename=path + '/run_all.log', filemode='w', level=logging.DEBUG)
        # generate basic metadata
        logging.info('Processing dataset: %s', path)
        dataset_info = preprocessing(path, args.type, args.frame_rate, args.camera)
        logging.info(json.dumps(dataset_info))

        # run inference using multiple models 
        # run_inference(dataset_info, args.gpu_num)
        # run Vigil pipeline
        run_Vigil(dataset_info, 
                  gpu_num=args.gpu_num, 
                  local_model='Inception', 
                  profile_length=profile_length, 
                  segment_length=segment_length)
        
        # run Glimpse result
        run_Glimpse(dataset_info, profile_length, segment_length, target_f1)

    # path = os.path.join(babygroot_DT_ROOT, args.name)
    # print(path)
    # logging.basicConfig(filename=path + '/run_all.log', filemode='w', level=logging.DEBUG)
    # start_time = time.time()


    # # generate basic metadata
    # logging.info('Processing dataset: %s', path)
    # dataset_info = preprocessing(path, args.type, args.frame_rate, args.camera)
    # logging.info(json.dumps(dataset_info))

    # # run inference using multiple models 
    # run_inference(dataset_info, args.gpu_num)

    # # VideoStorm results
    # # run_VideoStorm(dataset_info, mode='e2e')
    # # run_VideoStorm(dataset_info, mode='motivation')

    # # AWStream results
    
    # #          

    # # Glimpse results

    # # Vigil results

    # run_Vigil(dataset_info, mode='e2e', gpu_num=args.gpu_num,local_model='Inception')
    # # run_Vigil(dataset_info, mode='motivation')

    

    print(time.time() - start_time)

if __name__ == '__main__':
    main()