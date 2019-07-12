import glob
import csv
import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import my_utils

VIDEOS = ['highway_no_traffic', 'reckless_driving','motor', 'jp_hw', 'russia', 'tw_road', 'tw_under_bridge']
# 'street_racing', 'traffic', 'highway_normal_traffic',

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
    path = '/home/zxxia/videos/'
    para_list = ['num_of_object', 'object_area', 'arrival_rate', 'velocity',
                 'total_object_area'] 

    colors = {}
    quants = Para_quantile()
    percent_frame_w_obj = []
    velocities = []
    obj_areas = []
    with open('stats.csv', 'w') as f:
        f.write('video name, num of objects, std, object area,std, '\
        'arrival rate, std, velocity,std, total area,std\n ')
        
        for video_name in VIDEOS:
            with open(path + video_name + '/metadata.json') as metadata_f:
                metadata = json.load(metadata_f)
            frame_cnt = metadata['frame count']
        #     dataset_name = os.path.basename(para_file).replace('Video_features_', '').replace('.csv','')
            para_file = path + video_name + "/profile/Video_features_"+video_name+'.csv'
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

            f.write(video_name + ',' + 
                    str(quants.num_of_object_quantile[video_name][2]) + ',' + 
                    str(quants.object_area_quantile[video_name][2]) + ',' +
                    str(quants.arrival_rate_quantile[video_name][2]) + ',' + 
                    str(quants.velocity_quantile[video_name][2]) + ',' + 
                    str(quants.total_object_area_quantile[video_name][2]) + '\n')

    plt.bar(VIDEOS, percent_frame_w_obj)
    plt.title("Percentage of frames with objects")
    plt.show()
    for velocity, video_name in zip(velocities, VIDEOS):
        #plot_cdf(velocity, video_name+' obj velocity cdf') 
        x, y = my_utils.CDF(velocity)
        plt.plot(x, y, label=video_name)
        axes = plt.gca()
        axes.set_xlim([-0.1, 10])
        axes.set_ylim([0, 1.1])
    plt.legend(loc='lower right')
    plt.title("Object Velocity CDF")
    plt.xlabel("Object Velocity")
    plt.ylabel("CDF")
    plt.show()

    for obj_area, video_name in zip(obj_areas, VIDEOS):
        x, y = my_utils.CDF(obj_area)
        plt.plot(x, y, label=video_name)
        axes = plt.gca()
        axes.set_xlim([-0.1, 0.8])
    plt.legend(loc='lower right')
    plt.title("Object Area CDF")
    plt.xlabel("Object Area")
    plt.ylabel('CDF')
    plt.show()

if __name__ == '__main__':
    main()


