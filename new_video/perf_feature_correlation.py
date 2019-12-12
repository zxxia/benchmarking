from collections import defaultdict
import os
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import numpy as np
import glob
import math
from utils.utils import load_metadata

def load_video_features(filename):
    video_clips = []
    obj_cnt = []
    obj_area = []
    obj_velocity = []
    arriv_rate = []
    tot_area = []
    percent_frame_w_obj = []
    similarity = []
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            cols = line.strip().split(',')
            video_clips.append(cols[0])
            obj_cnt.append(float(cols[1]))
            obj_area.append(float(cols[3]))
            arriv_rate.append(float(cols[5]))
            obj_velocity.append(float(cols[7]))
            tot_area.append(float(cols[9]))
            percent_frame_w_obj.append(float(cols[11]))
            similarity.append(float(cols[12]))

    return video_clips, obj_cnt, obj_area, arriv_rate, obj_velocity, tot_area, \
           percent_frame_w_obj, similarity


def nonzero(orig_list):
    nonzero_list = [e for e in orig_list if e != 0]
    return nonzero_list

youtube_video_list = ['traffic', 'highway_normal_traffic' ]#'walking','driving_downtown','crossroad',
                    #'crossroad2','crossroad3', 'crossroad4','crossroad5',
                    #'driving1','driving2']
#youtube_video_list = ['highway','driving2']
gpu_time_per_frame = 100
def load_performance(perf_file):
    videos = []
    perf_list = []
    acc_list = []
    with open(perf_file, 'r') as f:
        f.readline()
        for line in f:
            line_list = line.strip().split(',')
            videos.append(line_list[0])
            perf_list.append(float(line_list[1]))

            if len(line_list) == 3:
                acc_list.append(float(line_list[2]))

    return videos, perf_list, acc_list 

# def load_glimpse_performance(filename):
#     with open(filename, 'r') as f:
#         for line in f:
#             print(line)
#     return None, None


# class VideoFeatures: 
#     def __init__(self): 
#         self.num_of_object = 0
#         self.object_area = 0
#         self.arrival_rate = 0
#         self.velocity = 0
#         self.total_object_area = 0

# def read_para_KITTI(para_file):
#     num_of_object = []
#     object_area = []
#     arrival_rate = []
#     velocity = []
#     total_object_area = []
#     with open(para_file, 'r') as para_f:
#         para_f.readline()
#         for line in para_f:
#             line_list = line.strip().split(',')
#             if line_list[1] != '':
#                 num_of_object.append(int(line_list[1]))
#             else:
#                 num_of_object.append(0)
#             if line_list[2] != '':
#                 object_area +=  [float(x) for x in line_list[2].split(' ')]
#             else:
#                 object_area.append(0)
#             if line_list[3] != '':
#                 arrival_rate.append(int(line_list[1]))
#             else:
#                 arrival_rate.append(0)
#             if line_list[4] != '':
#                 velocity +=  [float(x) for x in line_list[4].split(' ')]
#             else:
#                 velocity.append(0)
#             if line_list[5] != '':
#                 total_object_area.append(float(line_list[5]))
#             else:
#                 total_object_area.append(0)
#     percent_of_frame_w_object = len(nonzero(num_of_object))/float(len(num_of_object))
#     return num_of_object, object_area, arrival_rate, velocity, total_object_area, percent_of_frame_w_object



def read_para(feature_file, metadata_file):
    num_of_object = defaultdict(list)
    object_area = defaultdict(list)
    arrival_rate = defaultdict(list)
    velocity = defaultdict(list)
    total_object_area = defaultdict(list)
    percent_of_frame_w_object = {}
    #para_file = para_path + '/Video_features_' + dataset_name + '.csv'
    metadata = load_metadata(metadata_file)
    frame_rate = metadata['frame rate']
    with open(feature_file, 'r') as para_f:
        para_f.readline()
        for line in para_f:
            line_list = line.strip().split(',')
            frame_id = int(line_list[0])
            chunk_length = 30 # chunk a long video into 30-second short videos
            key = str(frame_id // int(chunk_length*frame_rate))

            if line_list[1] != '':
                num_of_object[key].append(int(line_list[1]))
            else:
                num_of_object[key].append(0)
            if line_list[2] != '':
                object_area[key] +=  [float(x) for x in line_list[2].split(' ')]
            else:
                object_area[key].append(0)
            if line_list[3] != '':
                arrival_rate[key].append(int(line_list[1]))
            else:
                arrival_rate[key].append(0)
            if line_list[4] != '':
                velocity[key] +=  [float(x) for x in line_list[4].split(' ') if float(x) != 100]
            else:
                velocity[key].append(0)
            if line_list[5] != '':
                total_object_area[key].append(float(line_list[5]))
            else:
                total_object_area[key].append(0)

            percent_of_frame_w_object[key] = len(nonzero(num_of_object[key]))/float(len(num_of_object[key]))

    return num_of_object, object_area, arrival_rate, velocity, total_object_area, percent_of_frame_w_object


def load_video_feature(para_path, metadata_file):
    features = {}
    # for para_file in glob.glob(para_path + '/Video_features_KITTI*'):
    #     dataset_name =format(int(para_file.split('_')[-1].replace('.csv','')), '04d')    
    #     VideoFeatures = read_para_KITTI(para_file)
    #     features[dataset_name] = [np.median(nonzero(x)) for x in VideoFeatures[:-1]]
    #     features[dataset_name].append(VideoFeatures[-1])


    for dataset_name in youtube_video_list:
        VideoFeatures = read_para(para_path, metadata_file)

        for chunk_index in VideoFeatures[1].keys():
            features[chunk_index] =  [np.median(nonzero(x[chunk_index])) for x in VideoFeatures[:-1]]
            features[chunk_index].append(VideoFeatures[-1][chunk_index])
                # [np.median(nonzero(num_of_object_per_chunk[chunk_index])),
                #  np.median(nonzero(object_area_per_chunk[chunk_index])),
                #  np.median(nonzero(arrival_rate_per_chunk[chunk_index])),
                #  np.median(nonzero(velocity_per_chunk[chunk_index])),
                #  np.median(nonzero(total_object_area_per_chunk[chunk_index]))]
    return features
# 
# 
# def load_video_feature2(para_path):
#     features = {}
#     # for para_file in glob.glob(para_path + '/Video_features_KITTI*'):
#     #     dataset_name =format(int(para_file.split('_')[-1].replace('.csv','')), '04d')    
#     #     VideoFeatures = read_para_KITTI(para_file)
#     #     features[dataset_name] = [np.median(nonzero(x)) for x in VideoFeatures[:-1]]
#     #     features[dataset_name].append(VideoFeatures[-1])
#     with open(para_path, 'r') as f:
#         for line in f:
#             line_list = line.strip().split(',')
#             dataset_name = line_list[0]
#             features[dataset_name] = [float(x) for x in line_list[2:]]
#     return features

def main():
    dataset_name = []
    perf_dict = {}
    feature_names = ['Object Count',
                     'Object Size',
                     'Arrival Rate',
                     'Object Velocity',
                     'Total Object Area',
                     'Percent of Frame with Object']
    # perf_file = '/Users/zhujunxiao/Desktop/benchmarking/Final_code/videostorm/data/VideoStorm_Performance_KITTI.csv'    
    # perf_dict, dataset_name = load_performance(perf_file, 
    #                                            perf_dict, 
    #                                            dataset_name)
    perf_file = '/home/zxxia/benchmarking_orig/videostorm/videostorm_result.csv'

    perf_list = load_performance(perf_file)

    # features = load_video_feature(para_path)
    para_path = '/home/zxxia/benchmarking_orig/new_video/stats_zxxia.csv'
    #metadata_file = '/mnt/data/zhujun/new_video/jp/metadata.json'
    features = load_video_features(para_path) # , metadata_file)

    for i, feature in enumerate(features):
        #print(i, len(feature), len(perf_list))
        plt.figure(i)
        plt.scatter(feature, perf_list)
        plt.title('videostorm vs '+feature_names[i])
        plt.xlabel(feature_names[i])
        plt.ylabel('GPU processing time')
    
    plt.legend()
    plt.show()
    # for i in range(0, 6):
    #     VS_perf_vec = []
    #     feature_vec = []
    #     fig, ax = plt.subplots(1,1, sharex=True)
    #     for key in perf_dict.keys():
    #         if key not in features: 
    #             print('feature not found:', key)
    #             continue
    #         if math.isnan(features[key][i]):
    #             continue
    #         VS_perf_vec.append(perf_dict[key])
    #         feature_vec.append(features[key][i])
    #         ax.scatter(features[key][i], perf_dict[key],c='r')

    #         # ax.annotate(key, (features[key][i], perf_dict[key]))
    #         # ax.scatter(features[key][i], ,c='b')
    #     coef, p = spearmanr(VS_perf_vec, feature_vec)
    #     alpha = 0.05
    #     if p > alpha:
    #         pass
    #     else:            
    #         print(feature_names[i])
    #         print('Spearmans correlation coefficient: %.3f' % coef)
    #         print('Samples are correlated (reject H0) p=%.3f' % p)    
    #     plt.xlabel(feature_names[i])
    #     plt.ylabel('GPU processing time (ms)')
    #     #plt.xlim([1,4])
    #     plt.show()
    #     # print(features[key])

    # with open('test.csv', 'w') as f:
    #     for key in sorted(perf_dict.keys()):
    #         if key not in features: 
    #             print('feature not found:', key)
    #             continue

    #         f.write(key + ',' + str(perf_dict[key])+ ',' 
    #             +','.join([str(x) for x in features[key]]) + '\n')
    # return 

if __name__ == '__main__':
    main()
