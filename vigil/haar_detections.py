""" get haar detections  """
import argparse
# import pdb
import cv2
from Vigil.helpers import convert_box_coordinate
from utils.utils import load_metadata


SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 10
PIXEL_THRESHOLD = 0.5

SCALE_FACTOR_DICT = {
    'traffic': 1.01,
    'jp_hw': 1.01,
    'russia': 1.01,
    'tw_road': 1.01,
    'tw_under_bridge': 1.01,
    'nyc': 1.01,
    'lane_split': 1.01,
    'tw': 1.01,
    'tw1': 1.01,
    'russia1': 1.01,
    'park': 1.01,
    'drift': 1.01,
    'crossroad3': 1.01,
    'crossroad2': 1.01,
    'crossroad': 1.01,
    'driving2': 1.01,
    'crossroad4': 1.01,
    'driving1': 1.01,
    'driving_downtown': 1.01,
    'highway': 1.01,
    'highway_normal_traffic': 1.01,
    'jp': 1.03,
    'motorway': 1.01,
}

MIN_NEIGHBORS_DICT = {
    'traffic': 1,
    'jp_hw': 1,
    'russia': 1,
    'tw_road': 1,
    'tw_under_bridge': 1,
    'nyc': 1,
    'lane_split': 1,
    'tw': 1,
    'tw1': 1,
    'russia1': 1,
    'park': 1,
    'drift': 1,
    'crossroad3': 1,
    'crossroad2': 1,
    'crossroad': 1,
    'driving2': 1,
    'crossroad4': 1,
    'driving1': 1,
    'driving_downtown': 1,
    'highway': 1,
    'highway_normal_traffic': 1,
    'jp': 1,
    'motorway': 1,
}


def change_box_format(boxes):
    """ Change box format from [x, y, w, h] to [xmin, ymin, xmax, ymax] """
    return [convert_box_coordinate(box) for box in boxes]


def haar_detection(img, car_cascade, dataset):
    """ img: opencv image
        car_cascade
        return list of bounding boxes. box format: [xmin, ymin, xmax, ymax]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray)

    # cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    cars = car_cascade.detectMultiScale(gray, SCALE_FACTOR_DICT[dataset],
                                        MIN_NEIGHBORS_DICT[dataset])

    cars = change_box_format(cars)
    return cars


def main():
    """ haar detection """
    parser = argparse.ArgumentParser(description="get haar detection results")
    parser.add_argument("--path", type=str, help="path contains all datasets")
    parser.add_argument("--video", type=str, help="video name")
    parser.add_argument("--metadata", type=str, default='',
                        help="metadata file in Json")
    parser.add_argument("--output", type=str, help="output result file")

    args = parser.parse_args()
    path = args.path
    dataset = args.video
    output_file = args.output
    metadata_file = args.metadata

    metadata = load_metadata(metadata_file)
    frame_cnt = metadata['frame count']

    cascade_src = 'haar_models/cars.xml'
    car_cascade = cv2.CascadeClassifier(cascade_src)

    with open(output_file, 'w', 1) as f_out:
        f_out.write('video, f1, bw, scale factor, min neighbors, '
                    'avg upload area, avg total obj area\n')
        for img_idx in range(1, frame_cnt + 1):
            img_name = '{:06d}.jpg'.format(img_idx)
            img = cv2. imread(path + dataset + '/720p/'+img_name)

            # get simple proposed regions which might have objects
            for box in haar_detection(img, car_cascade, dataset):
                f_out.write(','.join([str(img_idx), str(box[0]),
                                      str(box[1]), str(box[2]),
                                      str(box[3])])+'\n')


if __name__ == '__main__':
    main()
