import glob
import copy
import csv
import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import my_utils
import pdb
from statistics import median

PATH = '/data/zxxia/videos/'
#para_list = ['num_of_object', 'object_area', 'arrival_rate', 'velocity',
#             'total_object_area'] 

VIDEOS = sorted(['traffic', 'reckless_driving','motor', 'jp_hw', 'russia', 'tw_road', 
          'tw_under_bridge', 'highway_normal_traffic', 'highway_no_traffic',
          'tw', 'tw1', 'jp', 'russia', 'russia1','drift', 'park'])

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
                    object_area +=  [float(x) for x in line_list[2].split(' ')]
                else:
                    object_area.append(0)

                if line_list[3] != '':
                    arrival_rate.append(int(line_list[1]))
                else:
                    arrival_rate.append(0)

                if line_list[4] != '':
                    velocity +=  [float(x) for x in line_list[4].split(' ')]
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
    return num_of_object, object_area, arrival_rate, velocity, total_object_area, frame_with_obj_cnt

def parse_feature_line(line):
    features = dict()
    frame_id, obj_cnt, obj_areas, arrival_rate, velocities, tot_obj_area, \
    obj_types, dominant_type = line.strip().split(',')
    if frame_id == '':
        features['frame id'] = 0
    else: 
        features['frame id'] = int(frame_id)

    if obj_cnt == '':
        features['object count'] = 0
    else:
        features['object count'] = int(obj_cnt)
    if obj_areas == '':
        features['object area'] = []
    else:
        features['object area'] = [float(x) for x in obj_areas.split(' ')]

    if arrival_rate == '':
        features['arrival rate'] = 0
    else:
        features['arrival rate'] = int(arrival_rate)
    if velocities == '':
        features['velocity'] = []
    else:
        features['velocity'] =  [float(x) for x in velocities.split(' ')]

    if tot_obj_area == '':
        features['total object area'] = 0
    else:
        features['total object area'] = float(tot_obj_area)
                
    features['object types'] = obj_types.split(' ')
    features['dominant type'] = dominant_type

    return features


def parse_feature_file(filename, frame_rate, interval):
    interval_frame_cnt = int(frame_rate * interval)
    video_features = list() 

    chunk_features = defaultdict(list)
    frame_w_obj = 0
    with open(filename, 'r') as f:
        f.readline() # drop header 
        for line in f:
            #pdb.set_trace()
            features = parse_feature_line(line)
            if features['frame id'] % (interval_frame_cnt + 1) == 0:
                chunk_features['frame w/ obj percent'] = frame_w_obj / interval_frame_cnt
                video_features.append(chunk_features) 
                chunk_features = defaultdict(list)
                frame_w_obj = 0

            chunk_features['object count'].append(features['object count'])
            chunk_features['object area'].extend(features['object area'])
            chunk_features['arrival rate'].append(features['arrival rate'])
            chunk_features['velocity'].extend(features['velocity'])
            chunk_features['total object area'].append(features['total object area'])
            chunk_features['object types'].extend(features['object types'])
            chunk_features['dominant type'].append(features['dominant type'])
            if '3' in features['object types'] or '8' in features['object types']:
                frame_w_obj += 1
        chunk_features['frame w/ obj percent'] = frame_w_obj / features['frame id'] % (interval_frame_cnt + 1)
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
    fig, ax = plt.subplots(1,1, sharex=True)
    ax.boxplot(quant.values(), 
    showfliers=False, patch_artist=True)
    plt.xticks(np.arange(1, 1+len(quant.keys())), 
    quant.keys())
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

    colors = {}
    quants = Para_quantile()
    percent_frame_w_obj = []
    velocities = []
    obj_areas = []
    with open('stats.csv', 'w') as f:
        f.write('video name,num of objects,std,object area,std, '\
                'arrival rate,std,velocity,std,total area,std,frame w/ obj percent\n')
        
        for video_name in VIDEOS:
            metadata = my_utils.load_metadata(PATH + video_name + '/metadata.json')
            frame_cnt = metadata['frame count']
            frame_rate = metadata['frame rate']
            #     dataset_name = os.path.basename(para_file).replace('Video_features_', '').replace('.csv','')
            para_file = PATH + video_name + "/profile/Video_features_" + video_name+'.csv'
            features = parse_feature_file(para_file, frame_rate, 30)
            for i, ft in enumerate(features):
                f.write(video_name + '_' + str(i) +',' + 
                        str(median(ft['object count'])) + ',' + 
                        str(np.std(ft['object count'])) + ',' +
                        str(median(ft['object area'])) + ',' + 
                        str(np.std(ft['object area'])) + ',' +
                        str(median(ft['arrival rate'])) + ',' + 
                        str(np.std(ft['arrival rate'])) + ',' + 
                        str(median(ft['velocity'])) + ',' + 
                        str(np.std(ft['velocity'])) + ',' + 
                        str(median(ft['total object area'])) + ','+
                        str(np.std(ft['total object area'])) + ','+
                        str(ft['frame w/ obj percent']) + '\n')

   
            print(para_file)
            num_of_object, object_area, arrival_rate, velocity, total_object_area, frame_with_obj_cnt = read_para(para_file)
            velocities.append(nonzero(velocity))
            obj_areas.append(object_area)
            #plot_cdf(velocity) 
            #print(type(velocity))
            #print(frame_with_obj_cnt)
            percent_frame_w_obj.append(float(frame_with_obj_cnt)/float(frame_cnt))

            quants.num_of_object_quantile[video_name] = compute_quantile(num_of_object)
            quants.object_area_quantile[video_name] = compute_quantile(object_area)
            quants.arrival_rate_quantile[video_name] = compute_quantile(arrival_rate)
            quants.velocity_quantile[video_name] = compute_quantile(velocity)                      
            quants.total_object_area_quantile[video_name] = compute_quantile(total_object_area)

            # f.write(video_name + ',' + 
            #         str(quants.num_of_object_quantile[video_name][2]) + ',' + 
            #         str(quants.object_area_quantile[video_name][2]) + ',' +
            #         str(quants.arrival_rate_quantile[video_name][2]) + ',' + 
            #         str(quants.velocity_quantile[video_name][2]) + ',' + 
            #         str(quants.total_object_area_quantile[video_name][2]) + '\n')

#    plt.bar(VIDEOS, percent_frame_w_obj)
#    plt.title("Percentage of frames with objects")
#    plt.show()
#    for velocity, video_name in zip(velocities, VIDEOS):
#        #plot_cdf(velocity, video_name+' obj velocity cdf') 
#        x, y = my_utils.CDF(velocity)
#        plt.plot(x, y, label=video_name)
#        axes = plt.gca()
#        axes.set_xlim([-0.1, 10])
#        axes.set_ylim([0, 1.1])
#    plt.legend(loc='lower right')
#    plt.title("Object Velocity CDF")
#    plt.xlabel("Object Velocity")
#    plt.ylabel("CDF")
#    plt.show()
#
#    for obj_area, video_name in zip(obj_areas, VIDEOS):
#        x, y = my_utils.CDF(obj_area)
#        plt.plot(x, y, label=video_name)
#        axes = plt.gca()
#        axes.set_xlim([-0.1, 0.8])
#    plt.legend(loc='lower right')
#    plt.title("Object Area CDF")
#    plt.xlabel("Object Area")
#    plt.ylabel('CDF')
#    plt.show()

if __name__ == '__main__':
    main()


