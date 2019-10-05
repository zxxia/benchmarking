import pdb
from videostorm.profiler import profile, profile_eval
from utils.model_utils import load_ssd_detection, load_kitti_ground_truth

PATH = '/mnt/data/zhujun/dataset/KITTI/'
TEMPORAL_SAMPLING_LIST = [20, 15, 10, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2, 1]
MODEL_LIST = ['FasterRCNN']  # MODEL_LIST = ['FasterRCNN','SSD']
DATASET_LIST = ['City', 'Residential', 'Road']

TARGET_F1 = 0.9
OFFSET = 0  # The time offset from the start of the video. Unit: seconds
CHUNK_LENGTH = 30  # A long video is chopped into chunks. Unit: second
PROFILE_LENGTH = 10  # Profiling length within a chunk. Unit: second
KITTI_FRAME_RATE = 10
VIDEO_INDEX_DICT = {'City': [1, 2, 5, 9, 11, 13, 14, 17, 18, 48, 51, 56, 57, 59, 60, 84, 91, 93],
                    'Road': [15, 27, 28, 29, 32, 52, 70],
                    'Residential': [19, 20, 22, 23, 35, 36, 39, 46, 61, 64, 79, 86, 87]}


def main():
    with open('videostorm_coverage_kitti_motivation.csv', 'w') as f:
        f.write("video_name,frame_rate,f1\n")
        for dataset in DATASET_LIST:
            print("processing", dataset)
            frame_rate = KITTI_FRAME_RATE
            # profiling
            for i in VIDEO_INDEX_DICT[dataset]:
                gt_file = PATH + dataset + '/2011_09_26_drive_' + \
                    format(i, '04d') + \
                    '_sync/Parsed_ground_truth.csv'
                print(gt_file)
                clip = dataset+'_'+str(i)
                gt = load_kitti_ground_truth(gt_file)
                # print(gt)
                start_frame = min(gt.keys())
                end_frame = max(gt.keys())
                duration = (end_frame-start_frame) / frame_rate
                if duration < 10.0:
                    continue
                # pdb.set_trace()

                profile_start = start_frame
                profile_end = int((end_frame - start_frame)/3 + start_frame)
                print('profile {} {} {}'.format(clip, profile_start, profile_end))

                best_frame_rate = profile(gt, gt, profile_start, profile_end,
                                          frame_rate, TEMPORAL_SAMPLING_LIST)
                print(best_frame_rate)

                test_start = 1 + profile_end
                test_end = end_frame
                # load fasterRCNN + full resolution +
                # highest frame rate as ground truth

                # test on the rest of the short video
                print('test {} {} {}'.format(clip, test_start, test_end))
                best_sample_rate = frame_rate/best_frame_rate
                f1 = profile_eval(gt, gt, frame_rate, best_sample_rate,
                                  test_start, test_end)

                print(clip, best_frame_rate, f1)
                f.write(clip + ',' + str(1/best_sample_rate)
                        + ',' + str(f1) + '\n')


if __name__ == '__main__':
    main()
