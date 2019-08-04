from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from statistics import median
from perf_feature_correlation import load_performance, load_video_features
from scipy.stats import spearmanr, kendalltau
import pdb
NUM_OF_VIDEOS = 10
# SELECTED_VIDEOS = {
# 'Object Count': ['russia1_32', 'russia1_20', 'traffic_0' ,'russia1_2', 
#                  'tw1_2', 'russia_7', 'tw1_20', #'tw_road_4', 
#                  'russia1_27', 'tw1_7', 'jp_32'],
# 'Object Area': [ 'reckless_driving_2', 'reckless_driving_0', 'park1',
#                 'reckless_driving_6', 'reckless_driving_7', 'reckless_driving_8',
#                 'reckless_driving_4', 'reckless_driving_3', 'reckless_driving_1',
#                 'reckless_driving_5'], 
# 
# 'Object Velocity': ['russia1_6', 'tw_road_7', 'russia1_25', 'park_10 1', 
#                     'park_12', 'tw_under_bridge_5',  'park_26', 
#                     'russia1_30', 'park_7'], 
# 
# 'Total Object Area': [ 'motor_4', 'motor_3', 'jp_hw_9',
#                       'tw_39', 'russia1_3', 'tw1_1',  'motor_1', 'motor_2'],
# 'Percent of Frame with Object': ['russia1_22', 'russia1_25', 'park_2', 
#                                  'highway_no_traffic_0', 'russia1_34', 
#                                  'park_38', 'tw_under_bridge_5', 'park_16', 
#                                  'highway_no_traffic_3']#, 'reckless_driving_6'] 
# }
# 
# 
# SELECTED_VIDEOS = {
# 'Object Count': ['tw1_4', 'jp_8', 'jp_6', 'russia1_5', 'jp_7', 'tw_6', 'jp_3', 'tw_9', 'tw1_7', 'jp_2'], # back up tw_0, tw_5, sort object velocity
# 'Object Velocity': ['drift_3', 'highway_no_traffic_2', 'drift_6', 'park_4', 
#                    'drift_4', 'highway_no_traffic_1', 'drift_0', 'highway_no_traffic_0', 'park_0', 'park_6', 'drift_5'], # sort % frame with obj
# 'Object Area': ['drift_3'], # useless
# 'Total Object Area': ['drift_3'], # useless
# 'Percent of Frame with Object': ['motor_1', 'drift_5', 'park_3', 'drift_6', 'drift_9', 'drift_8', 'drift_0', 'drift_7', 'drift_4', 'drift_2'] #sort on object velocity
# }

# SELECTED_VIDEOS = {
# 'Object Count' : [  'russia1_0', 'crossroad2_15', 'crossroad2_16',
# 'crossroad2_0', 'crossroad2_20', 'crossroad2_17', 'crossroad2_21',
#  'tw_2', 'crossroad2_19','tw_road_1','russia_1',], #
# 'Object Area' : [ 'walking_9', 'nyc_0', 'walking_7', 'driving1_0',
#                   'tw_road_0', 'russia_1', 'walking_13', 'driving1_6',
#                   'walking_1', ],
# 'Object Velocity': ['highway_6', 'crossroad3_0', 'crossroad4_0',
#                     'highway_5', 'highway_7', 'highway_9', 'highway_4',
#                     'highway_0', 'highway_1',],
# 'Percent of Frame with Object' : [ 'driving1_0', 'driving1_1', 'driving1_2',
#                                    'driving1_7',  'driving2_7',
#                                    'russia1_1','walking_15']
# }

SELECTED_VIDEOS = {
'Object Count' : ['crossroad_12', 'crossroad2_6', 'crossroad2_7', 'crossroad2_8',
                  'crossroad2_10','crossroad2_12', 'crossroad2_14', 'crossroad2_18',
                  'crossroad2_20','crossroad2_21','crossroad3_0','crossroad3_1',
                  'crossroad3_2','crossroad3_6'], #
'Object Area' : ['nyc_1', 'tw_road_0', 'crossroad_9', 'nyc_4', 'tw_7', 'crossroad_8', 'tw_2', 'nyc_0', 'nyc_5', 'nyc_7', 'tw1_5'],

'Object Velocity': ['crossroad_1', 'crossroad_9', 'crossroad2_4', 'crossroad2_7',
                    'crossroad3_1', 'highway_2', 'highway_5', 'highway_8', 
                    'highway_10', 'highway_11', 'highway_15'],
'Percent of Frame with Object' : ['driving1_1','driving1_2','driving2_0','highway_0','highway_12','highway_14','lane_split_0','park_1','park_5','tw_under_bridge_0']}
# ['walking_17','driving1_3','driving2_3',
#                                   'russia1_3','russia1_4','russia1_7',
#                                   'tw1_8','walking_3','walking_9','walking_10',
#                                   'walking_11','walking_12'] }

def load_awstream_results(filename):
    video_clips = []
    f1 = []
    bandwidth = []
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            cols = line.strip().split(',')
            video_clips.append(cols[0])
            f1.append(float(cols[2]))
            bandwidth.append(float(cols[4]))

    return video_clips, bandwidth, f1


def compute_cv(video_list, perf_list, selected_videos):
    ft_to_perf = defaultdict(list)
    for video_name, perf in zip(video_list, perf_list):
        for key in selected_videos.keys():
            if video_name in selected_videos[key]:
                ft_to_perf[key].append(perf)
    coeff_var = [np.std(ft_to_perf[key])/np.mean(ft_to_perf[key]) 
                 for key in sorted(ft_to_perf.keys())]
    print(ft_to_perf)
    return coeff_var


def main():
    features_names = ['Object Count', 'Object Area', 
                      'Object Velocity',  
                      'Percent of Frame with Object']  #'Arrival Rate','Total Object Area',

    vs_perf_file = '/home/zxxia/benchmarking_orig/videostorm/videostorm_motivation_result.csv'
    vs_perf_video_list, vs_perf_list, vs_acc_list = load_performance(vs_perf_file)
    
    print(len(vs_perf_video_list), len(vs_perf_list))
    gl_perf_file = '/home/zxxia/benchmarking_orig/glimpse/glimpse_result.csv'
    gl_perf_video_list, gl_perf_list, gl_acc_list = load_performance(gl_perf_file)
    print(len(gl_perf_video_list), len(gl_perf_list))

    aw_perf_file = '/home/zxxia/benchmarking_orig/awstream/awstream_motivation_result.csv'

    aw_video_list, aw_bw_list, aw_acc_list = load_awstream_results(aw_perf_file)

    print('aw len', len(aw_video_list))
    video_clips, obj_cnt, obj_area, arrival_rate, obj_velocity, tot_obj_area, \
    percent_frame_w_obj, similarity = load_video_features('stats.csv') 
    print(len(video_clips))

    vs_coeff_var = compute_cv(vs_perf_video_list, vs_perf_list, SELECTED_VIDEOS)
    vs_coeff_var_f1 = compute_cv(vs_perf_video_list, vs_acc_list, SELECTED_VIDEOS)
    gl_coeff_var = compute_cv(gl_perf_video_list, gl_perf_list, SELECTED_VIDEOS)
    
    aw_coeff_var = compute_cv(aw_video_list, aw_bw_list, SELECTED_VIDEOS)
    print('aw coeff var', aw_coeff_var)
    aw_coeff_var_f1 = compute_cv(aw_video_list, aw_acc_list, SELECTED_VIDEOS)
    #print(len(vs_coeff_var))
    #print(vs_coeff_var)
    # Section for plot the motivation bar plot
    plt.figure('Feature to Perf')
    width = 0.35
    ind = np.arange(len(features_names))
    plt.bar(ind - width/2, vs_coeff_var, width, label='videostorm performance')
    plt.bar(ind - width/2, vs_coeff_var_f1, width, bottom=vs_coeff_var, label='videostorm accuracy')

    plt.bar(ind + width/2, aw_coeff_var, width, label='awstream bandwidth')
    plt.bar(ind + width/2, aw_coeff_var_f1, width, bottom=aw_coeff_var, label='awstream accuracy')
    #plt.bar(ind + width/2, gl_coeff_var, width, label='glimpse')
    plt.gca().set(xticks=ind, xticklabels=sorted(features_names),
                  ylabel='Coefficent of Variantion')
    plt.legend()
    plt.show()
    return


    # Code for basic statistics of the original feature values
    features = [obj_cnt, obj_area,  obj_velocity,  
                percent_frame_w_obj]#arrival_rate,tot_obj_area,
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print(np.corrcoef(features))
    # features = np.transpose(np.array(features).reshape((len(features),-1)))
    # 
    # features_mean = np.mean(features, axis=0)
    # features_std = np.std(features, axis=0)
    # features_range = np.ptp(features, axis=0)
    # features_median = np.median(features, axis=0)
    # features_max = np.max(features, axis=0)
    # features_min = np.min(features, axis=0)
    # print('features mean', features_mean)
    # print('features median', features_median)
    # print('features max', features_max)
    # print('features min', features_min)
    # print('features std', features_std)
    # print('features range', features_range)
    
    # results = list()
    # for i in range(len(video_clips)):
    #     ref_video = i
    #     result = list()
    #     #if similarity[i] > 0.5:
    #     #    continue
    #     result.append(i)
    #     for j in range(len(video_clips)):
    #         if j == i:
    #             continue

    #         if np.abs(obj_cnt[j] - obj_cnt[i]) <= (1 * features_range[0]) and \
    #            np.abs(obj_area[j] - obj_area[i]) <= (0.10 * features_range[1]) and \
    #            np.abs(obj_velocity[j] - obj_velocity[i]) <= (0.050 * features_range[2]) and \
    #            np.abs(percent_frame_w_obj[j] - percent_frame_w_obj[i]) <= 1 * features_range[3]:#\
    #             result.append(j) 
    #             # if similarity[i] > 0.5 and similarity[j] > 0.5:
    #             #     result.append(j)
    #             # if similarity[i] <= 0.5 and similarity[j] <= 0.5:
    #             #     result.append(j)
    #     if len(result) >1 and np.ptp(np.array(percent_frame_w_obj)[result])/features_range[3] >=0.70:
    #         results.append(result)
    #         print(len(result))
    # print('results length:', len(results))
    # with open('tmp.csv', 'w') as f:
    #     for result in results:
    #         for i in result:
    #             f.write(video_clips[i] + ',' + str(obj_cnt[i]) + ',' + 
    #                     str(obj_area[i]) + ',' + str(obj_velocity[i]) + ',' +
    #                     str(percent_frame_w_obj[i]) + ',' + str(similarity[i]) + ',' +
    #                     str(vs_perf_list[i]) + ',' + str(vs_acc_list[i]) + '\n')
    #         f.write('\n')
    # return

    # exp_ft_idx = 3
    # ctrl_ft_idx = 2
    # indices_sort_ctrl_ft = features[:, ctrl_ft_idx].argsort()
    # # print(features[indices_sort_contrl_ft][:10])

    # with open('target_{}.csv'.format(features_names[exp_ft_idx]), 'w') as f:
    #     f.write('video,object count,object area,object velocity,%frame with object,object count similarity,object area similarity,object velocity similarity,perf,f1\n')
    #     for i in np.arange(features.shape[0] - NUM_OF_VIDEOS):
    #         # print(np.any(np.array(obj_cnt_similarity)[indices_sort_ctrl_ft][i:i+NUM_OF_VIDEOS]>0.25))
    #         # if (np.any(np.array(obj_cnt_similarity)[indices_sort_ctrl_ft][i:i+NUM_OF_VIDEOS] > 0.5) \
    #         # or np.any(np.array(obj_area_similarity)[indices_sort_ctrl_ft][i:i+NUM_OF_VIDEOS] > 0.2) \
    #         # or np.any(np.array(obj_velocity_similarity)[indices_sort_ctrl_ft][i:i+NUM_OF_VIDEOS] > 0.2)):
    #         #     continue
    #         target_range = np.ptp(features[indices_sort_ctrl_ft][i:i+NUM_OF_VIDEOS, :], axis=0)
    #         range_percent = target_range / features_range
    #         mask = [True] * len(range_percent)
    #         mask[exp_ft_idx] = False
    #         if range_percent[exp_ft_idx] >= 0.8 and range_percent[2] <=0.1:
    #             print(range_percent, i, i+NUM_OF_VIDEOS, indices_sort_ctrl_ft[i:i+NUM_OF_VIDEOS])
    #             output_video_names = np.array(video_clips)[indices_sort_ctrl_ft][i:i+NUM_OF_VIDEOS].reshape(-1,1)
    #             similarities = np.hstack([np.array(obj_cnt_similarity).reshape(-1,1), 
    #                            np.array(obj_area_similarity).reshape(-1,1), 
    #                            np.array(obj_velocity_similarity).reshape(-1,1)])[indices_sort_ctrl_ft][i:i+NUM_OF_VIDEOS]
    #             print(output_video_names.shape)
    #             output_video_features = features[indices_sort_ctrl_ft][i: i+NUM_OF_VIDEOS]
    #             print(output_video_features.shape)
    #             vs_list = np.hstack([#np.array(vs_perf_video_list).reshape(-1,1), 
    #                                 np.array(vs_perf_list).reshape(-1,1), 
    #                                 np.array(vs_acc_list).reshape(-1,1)])[indices_sort_ctrl_ft][i:i+NUM_OF_VIDEOS]
    #             output_arr = np.column_stack((output_video_names, output_video_features, similarities,vs_list))
    #             # print(output_arr)
    #             print(output_arr.shape)
    #             np.savetxt(f,output_arr, fmt='%s',delimiter=',')
    #             f.write('\n')
    #         
    #     #print(len(perf_list), len(obj_cnt)) 
    # return


    for exp_ft_name, exp_ft in zip(features_names, features):
        print('Vary', exp_ft_name)
#         condition = np.ones(len(obj_cnt), dtype=bool)
        for ctrl_ft_name, ctrl_ft in zip(features_names, features):
            if ctrl_ft == exp_ft:
                continue

            plt.figure('Vary {},Control {}'.format(exp_ft_name, ctrl_ft_name))
            plt.hist(ctrl_ft, bins=50, alpha=0.6, label='Original')
            for i, video_clip in enumerate(video_clips):
                if video_clip in SELECTED_VIDEOS[exp_ft_name]:
                    plt.axvline(ctrl_ft[i], color='k', linestyle='dashed', linewidth=1)
                plt.gca().set(title='Vary {},Control {}'.format(exp_ft_name, ctrl_ft_name), 
                           xlabel=ctrl_ft_name, ylabel='Frequency')
            plt.savefig('/home/zxxia/figs/selected_videos/Vary_{}_Control_{}'.format(exp_ft_name, ctrl_ft_name)+'.png')
        
        # Plot disitrbutions of the original experimental feature and filtered 
        # experimental feature 
        plt.figure(exp_ft_name) 
        plt.hist(exp_ft, bins=50, alpha=0.6, label='Original')
        for i, video_clip in enumerate(video_clips):
            if video_clip in SELECTED_VIDEOS[exp_ft_name]:
                plt.axvline(exp_ft[i], color='k', linestyle='dashed', linewidth=1)
        plt.gca().set(title='Vary '+exp_ft_name, xlabel=exp_ft_name, 
                      ylabel='Frequency')
        plt.savefig('/home/zxxia/figs/selected_videos/' + 'Vary_'+exp_ft_name+'.png')
        
        # plt.legend()
        # _, max_ = plt.ylim()
        # plt.text(np.median(exp_ft), max_-max_/10, 'Median: {:.2f}'.format(np.median(exp_ft)))

        # for ctrl_ft_name, ctrl_ft in zip(features_names, features):
        #     if ctrl_ft == exp_ft:
        #         continue
        #     plt.figure('Vary {},Control {}'.format(exp_ft_name, ctrl_ft_name))
        #     plt.hist(ctrl_ft, bins=50, alpha=0.6, label='Original')
        #     plt.hist(np.array(ctrl_ft)[result_idx], bins=50, alpha=0.6, label='Filtered')
        #     #plt.axvline(np.median(exp_ft), color='k', linestyle='dashed', linewidth=1)
        #     plt.gca().set(title='Frequency Histogram', 
        #                   xlabel=ctrl_ft_name, ylabel='Frequency')
        #     plt.legend()
        # plt.show()





        # result_perf_list = np.array(perf_list)[result_idx]
        # result_exp_ft = np.array(exp_ft)[result_idx]
        # coef, p = spearmanr(result_perf_list, result_exp_ft) 

        # plt.figure(exp_ft_name + 'vs videostrom performance')
        # plt.scatter(result_exp_ft, result_perf_list)
        # plt.xlabel(exp_ft_name)
        # plt.ylabel('GPU Processing Time')
        # plt.title('rho={0:.3f}, p={1:.3f}'.format(coef, p))
        # plt.ylim(0,1)

        #jprint(max(np.array(exp_ft)[result_idx]), min(np.array(exp_ft)[result_idx]))
        #print(max(exp_ft), min(exp_ft))
        #print(np.ravel(np.array(video_clips)[result_idx]))

        #   compute the target feature median and std
        #   for the rest of the features
            
        # exp_ft_median = np.median(exp_ft) 
        #print(exp_ft_median)
        # exp_ft_std = np.std(exp_ft)
        # condition = np.logical_or(exp_ft >= (exp_ft_median + exp_ft_std),exp_ft <= (exp_ft_median - exp_ft_std))

    plt.show() 
    

if __name__ == '__main__':
    main()
