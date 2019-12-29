import csv
from benchmarking.chameleon.Chameleon import Chameleon, load_ground_truth, \
    load_model_predictions


MODEL_LIST = ['Inception', 'mobilenet', 'resnet50']
PROFILE_LENGTH = 10
SHORT_VIDEO_LENGTH = 30
OFFSET = 0


def main():
    pipeline = Chameleon(MODEL_LIST)
    for dataset in ['cropped_crossroad4', 'cropped_crossroad4_2',
                    'cropped_crossroad5', 'cropped_driving2']:
        gt_file = '../model_pruning/label/' + dataset + \
            '_ground_truth_car_truck_separate.csv'
        gt = load_ground_truth(gt_file)
        nb_frames = len(gt)
        frame_rate = 30
        chunk_frame_cnt = SHORT_VIDEO_LENGTH * frame_rate
        profile_frame_cnt = PROFILE_LENGTH * frame_rate
        nb_short_videos = (nb_frames - OFFSET * frame_rate) // chunk_frame_cnt
        with open('chameleon_e2e_result_{}.csv'.format(dataset), 'w', 1) as f:
            writer = csv.writer(f)
            writer.writerow(['video', 'model', 'gpu', 'f1'])
            for i in range(nb_short_videos):
                clip = dataset + '_' + str(i)
                start_frame = i * chunk_frame_cnt + 1 + OFFSET * frame_rate
                end_frame = (i + 1) * chunk_frame_cnt + OFFSET * frame_rate

                profile_start_frame = start_frame
                profile_end_frame = profile_start_frame + profile_frame_cnt - 1

                model_selection_file = '../model_pruning/label/' + \
                    dataset + '_model_predictions.csv'
                dt_dict = {}
                dt_dict['mobilenet'], dt_dict['Inception'], dt_dict['resnet50'] = \
                    load_model_predictions(model_selection_file)
                selected_model = pipeline.profile(
                    dt_dict, gt, [profile_start_frame, profile_end_frame])
                # print(selected_model

                test_start_frame = profile_end_frame + 1
                test_end_frame = end_frame
                gpu, f1 = pipeline.evaluate(dt_dict, selected_model, gt,
                                            [test_start_frame, test_end_frame])
                writer.writerow([clip, selected_model, gpu, f1])


if __name__ == '__main__':
    main()
