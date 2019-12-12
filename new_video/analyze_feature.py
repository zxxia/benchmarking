import glob
import copy
import csv
import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
from utils.utils import load_metadata
import pdb
from scipy import stats
from sklearn import preprocessing
import re
# from awstream.awstream import compress_images_to_video

PATH = '/mnt/data/zhujun/dataset/Youtube/'
para_list = ['Object Count', 'Object Size', 'Arrival Rate', 'Object Velocity', 'total_object_area']

VIDEOS = sorted(['walking',  'highway', 'crossroad2',
                 'crossroad', 'crossroad3', 'crossroad4',
                 'driving1', 'driving2', 'traffic',  'jp_hw', 'russia',
                 'tw_road', 'tw_under_bridge', 'highway_normal_traffic',
                 'tw', 'tw1', 'jp', 'russia1', 'drift', 'motorway',
                 'park', 'nyc', 'lane_split',  'driving_downtown'])
# , 'block', 'block1''reckless_driving',

CHUNK_LENGTH = 30  # 2*60 # seconds


def sample(v, sample_rate):
    return v[::sample_rate]


def nonzero(orig_list):
    nonzero_list = [e for e in orig_list if e != 0]
    return nonzero_list


def compute_quantile(data):
    quantile = []
    quantile.append(np.percentile(data, 5))
    quantile.append(np.percentile(data, 25))
    quantile.append(np.percentile(data, 50))
    quantile.append(np.percentile(data, 75))
    quantile.append(np.percentile(data, 95))
    return quantile


def read_para(para_file):
    num_of_object = []
    object_area = []
    arrival_rate = []
    velocity = []
    total_object_area = []
    frame_with_obj_cnt = 0
    try:
        with open(para_file, 'r') as para_f:
            para_f.readline()
            for line in para_f:
                line_list = line.strip().split(',')
                if line_list[1] != '':
                    num_of_object.append(int(line_list[1]))
                else:
                    num_of_object.append(0)

                if line_list[2] != '':
                    object_area += [float(x) for x in line_list[2].split(' ')]
                else:
                    object_area.append(0)

                if line_list[3] != '':
                    arrival_rate.append(int(line_list[1]))
                else:
                    arrival_rate.append(0)

                if line_list[4] != '':
                    velocity += [float(x) for x in line_list[4].split(' ')]
                else:
                    velocity.append(0)

                if line_list[5] != '':
                    total_object_area.append(float(line_list[5]))
                else:
                    total_object_area.append(0)

                if '3' in line_list[6] or '8' in line_list[6]:
                    frame_with_obj_cnt += 1
    except:
        print("File does not exist")
    return num_of_object, object_area, arrival_rate, velocity, \
           total_object_area, frame_with_obj_cnt


def parse_feature_line(line):
    features = dict()
    frame_id, obj_cnt, obj_area, arrival_rate, velocity, tot_obj_area, \
    obj_type,_ = line.strip().split(',')
    # , dominant_type
    if frame_id == '':
        frame_id = 0
    else:
        frame_id = int(frame_id)
    if obj_cnt == '':
        obj_cnt = 0
    else:
        obj_cnt = int(obj_cnt)
    if obj_area == '':
        obj_area = []
    else:
        obj_area = [float(x) for x in obj_area.split(' ')]

    if arrival_rate == '':
        arrival_rate = 0
    else:
        arrival_rate = int(arrival_rate)
    if velocity == '':
        velocity = []
    else:
        velocity = nonzero([float(x) for x in velocity.split(' ')])
    if tot_obj_area == '':
        tot_obj_area = 0
    else:
        tot_obj_area = float(tot_obj_area)
    if obj_type == '[]':
        obj_type = []
    else:
        obj_type = obj_type.split(' ')
    # if '3' not in features['object types'] or '8' not in features['object types']:
    #    print(features['object types'])
    # features['dominant type'] = dominant_type

    return frame_id, obj_cnt, obj_area, arrival_rate, velocity, tot_obj_area, \
        obj_type


def parse_feature_file(filename, frame_rate, video_frame_count, chunk_length):
    print(filename)
    chunk_frame_cnt = int(frame_rate * chunk_length)
    chunk_cnt = video_frame_count // chunk_frame_cnt
    chunk_idx = 0
    obj_cnt_profile, arrival_rate_profile, obj_area_profile, \
    obj_velocity_profile = [], [], [], []
    obj_cnt_test, arrival_rate_test, obj_area_test, \
    obj_velocity_test = [], [], [], []

    video_features = list()
    chunk_features = defaultdict(list)
    frame_w_obj = 0
    with open(filename, 'r') as f:

        #find = re.compile(r"(?:(?!profile).)*")
        #dataset_path = re.search(find, filename).group(0)
        #metadata = load_metadata(dataset_path + 'metadata.json')
        f.readline() # drop header
        for i, line in enumerate(f):
            frame_id, obj_cnt, obj_area, arrival_rate, \
            velocity, tot_obj_area, obj_type = parse_feature_line(line)

            if frame_id > (chunk_idx + 1) * chunk_frame_cnt:

                #compress_images_to_video(dataset_path,
                #                         chunk_idx * chunk_frame_cnt + 30 * frame_rate + 1,
                #                         chunk_frame_cnt - 30 * frame_rate,
                #                         metadata['frame rate'],
                #                         str(metadata['resolution'][0]) + 'x' + str(metadata['resolution'][1]),
                #                         25,
                #                         'tmp.mp4')
                #video_size1 = os.path.getsize('tmp.mp4')
                #os.remove("tmp.mp4")
                #compress_images_to_video(dataset_path+'540p',
                #                         chunk_idx * chunk_frame_cnt + 30 * frame_rate + 1,
                #                         chunk_frame_cnt - 30 * frame_rate,
                #                         metadata['frame rate'],
                #                         '960x540',
                #                         25,
                #                         'tmp.mp4')

                #video_size2 = os.path.getsize('tmp.mp4')
                #os.remove("tmp.mp4")
                #chunk_features['video size ratio'] = video_size1/video_size2


                chunk_features['object count similarity'] = 0  # stats.ks_2samp(obj_cnt_profile, obj_cnt_test)[0]
                chunk_features['object area similarity'] = 0 # stats.ks_2samp(obj_area_profile, obj_area_test)[0]
                if not obj_velocity_profile or not obj_velocity_test:
                    chunk_features['object velocity similarity'] = 1
                else:
                    chunk_features['object velocity similarity'] = 0  # stats.ks_2samp(obj_velocity_profile, obj_velocity_test)[0]
                chunk_features['arrival rate similarity'] = 0 # stats.ks_2samp(arrival_rate_profile, arrival_rate_test)[0]
                chunk_features['frame w/ obj percent'] = frame_w_obj / chunk_frame_cnt
                assert chunk_features['frame w/ obj percent'] <= 1
                video_features.append(chunk_features)
                chunk_idx += 1
                frame_w_obj = 0
                chunk_features = defaultdict(list)
                #print(frame_id, chunk_idx, chunk_cnt)
                obj_cnt_profile, arrival_rate_profile, obj_area_profile, \
                obj_velocity_profile = [], [], [], []
                obj_cnt_test, arrival_rate_test, obj_area_test, \
                obj_velocity_test = [], [], [], []

            chunk_features['object count'].append(obj_cnt)
            chunk_features['object area'].extend(obj_area)
            chunk_features['arrival rate'].append(arrival_rate)
            chunk_features['velocity'].extend(velocity)
            chunk_features['total object area'].append(tot_obj_area)
            chunk_features['object types'].extend(obj_type)
            if '3' in obj_type or '8' in obj_type:
                frame_w_obj += 1

            if frame_id <= (chunk_idx) * chunk_frame_cnt + 30 * frame_rate:
                obj_cnt_profile.append(obj_cnt)
                arrival_rate_profile.append(arrival_rate)
                obj_area_profile.extend(obj_area)
                obj_velocity_profile.extend(velocity)
            else:
                obj_cnt_test.append(obj_cnt)
                arrival_rate_test.append(arrival_rate)
                obj_area_test.extend(obj_area)
                obj_velocity_test.extend(velocity)

        if chunk_idx < chunk_cnt:
            # compress_images_to_video(dataset_path,
            #                          chunk_idx * chunk_frame_cnt + 30 * frame_rate + 1,
            #                          chunk_frame_cnt - 30 * frame_rate,
            #                          metadata['frame rate'],
            #                          str(metadata['resolution'][0]) + 'x' + str(metadata['resolution'][1]),
            #                          25,
            #                          'tmp.mp4')
            # video_size1 = os.path.getsize('tmp.mp4')
            # os.remove("tmp.mp4")
            # compress_images_to_video(dataset_path+'540p',
            #                          chunk_idx * chunk_frame_cnt + 30 * frame_rate + 1,
            #                          chunk_frame_cnt - 30 * frame_rate,
            #                          metadata['frame rate'],
            #                          '960x540',
            #                          25,
            #                          'tmp.mp4')

            # video_size2 = os.path.getsize('tmp.mp4')
            # os.remove("tmp.mp4")
            # chunk_features['video size ratio'] = video_size1/video_size2
            chunk_features['object count similarity'] = 0  # stats.ks_2samp(obj_cnt_profile, obj_cnt_test)[0]
            chunk_features['object area similarity'] = 0  # stats.ks_2samp(obj_area_profile, obj_area_test)[0]
            chunk_features['object velocity similarity'] = 0  # stats.ks_2samp(obj_velocity_profile, obj_velocity_test)[0]
            chunk_features['arrival rate similarity'] = 0  # stats.ks_2samp(arrival_rate_profile, arrival_rate_test)[0]
            chunk_features['frame w/ obj percent'] = frame_w_obj / chunk_frame_cnt
            assert chunk_features['frame w/ obj percent'] <= 1
            video_features.append(chunk_features)

    return video_features


class Para_quantile:
    def __init__(self):
        self.num_of_object_quantile = {}
        self.object_area_quantile = {}
        self.arrival_rate_quantile = {}
        self.velocity_quantile = {}
        self.total_object_area_quantile = {}


def plot_boxplot(quant, title):
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.boxplot(quant.values(), showfliers=False, patch_artist=True)
    plt.xticks(np.arange(1, 1+len(quant.keys())), quant.keys())
    plt.title(title)
    plt.show()


def plot_cdf(data, title):
    # Choose how many bins you want here
    num_bins = 20
    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=num_bins, density=True)
    dx = bin_edges[1] - bin_edges[0]
    # Now find the cdf
    cdf = np.cumsum(counts) * dx
    # And finally plot the cdf
    plt.plot(bin_edges[1:], cdf)
    plt.title(title)
    plt.show()


def main():
    with open('features_all_30s.csv', 'w') as f:
        f.write('video name,object count,object count std,object area,'
                'object area std,arrival rate,arrival rate std,'
                'object velocity,object velocity std,total area,'
                'total area std,frame with object percent,'
                'object count similarity,object area similarity,'
                'arrival rate similarity,object velocity similarity,'
                'video size ratio\n')

        for video_name in VIDEOS:
            metadata = load_metadata(PATH + video_name + '/metadata.json')
            frame_cnt = metadata['frame count']
            frame_rate = metadata['frame rate']
            resolution = metadata['resolution']
            para_file = PATH + video_name + '/' + str(resolution[1]) \
                + "p/profile/Video_features_" + video_name + ".csv"
            # pdb.set_trace()
            features = parse_feature_file(para_file, frame_rate, frame_cnt-1,
                                          CHUNK_LENGTH)
            for i, ft in enumerate(features):
                if ft['object count'] and ft['object area'] and \
                   ft['arrival rate'] and ft['velocity'] and \
                   ft['total object area'] and ft['frame w/ obj percent'] >= 0:

                    f.write(','.join([video_name+'_'+str(i),
                            str(np.median(nonzero(ft['object count']))),
                            str(np.std(nonzero(ft['object count']))),
                            str(np.median(nonzero(ft['object area']))),
                            str(np.std(nonzero(ft['object area']))),
                            str(np.median(nonzero(ft['arrival rate']))),
                            str(np.std(nonzero(ft['arrival rate']))),
                            str(np.median(ft['velocity'])),
                            str(np.std(ft['velocity'])),
                            str(np.median(nonzero(ft['total object area']))),
                            str(np.std(nonzero(ft['total object area']))),
                            str(ft['frame w/ obj percent']),
                            str(ft['object count similarity']),
                            str(ft['object area similarity']),
                            str(ft['arrival rate similarity']),
                            str(ft['object velocity similarity']),
                            str(ft['video size ratio']) + '\n']))


if __name__ == '__main__':
    main()
