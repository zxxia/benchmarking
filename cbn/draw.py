import csv
import matplotlib.pyplot as plt
import numpy as np

filename_car = ['highway(car)', 'london(car)', 'park(car)', 'motorway(car)']
#filename_car = ['london(car)']
filename_human = ['london(human)', 'tv_show(human)']

if __name__ == '__main__':
    fig = plt.figure()
    for list in [filename_human]:
        for car_name in list:
            Video_name = []
            velocity_avg = []
            file_velocity = './output/features_' + car_name + '.csv'
            with open(file_velocity, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    Video_name.append(row[0])
                    velocity_avg.append(row[44])
            Video_name = Video_name[1:]
            velocity_avg = [float(x) for x in velocity_avg[1:]]
            # print(Video_name)
            # print(velocity_avg)
            video_name = []
            gpu_time = []
            file_cost = './output/videostorm_results_' + car_name + '.csv'
            with open(file_cost, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    video_name.append(row[0])
                    gpu_time.append(row[2])

            video_name = video_name[1:]
            gpu_time = gpu_time[1:]
            print(gpu_time)
            gpu_time = [float(x) for x in gpu_time]

            assert len(Video_name) == len(video_name), "Error: different video_name length"

            plt.scatter(velocity_avg, gpu_time, label=car_name)
            plt.xlabel('Avg. object speed')
            plt.ylabel('Compute Cost(GPU)')
    plt.legend(loc="upper left")
    plt.title("Human feature VS VideoStorm performance")
    plt.show()




