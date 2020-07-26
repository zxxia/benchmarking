import sys  
sys.path.append('./') 
# print('\n'.join(sys.path))
"""Performance estimator module main function. 
This module will compute_features and use the pipeline profiler output to compute the estimated performance on the given video
The feature csv file will be output for user to double check."""


from performance_estimator.parser import parse_args
# from performance_estimator.run_pipeline import run_pipeline
# from performance_estimator.process_feature_performance import process_feature_performance
from feature_scanner.compute_features import *
from utils.utils import write_pickle_file,read_pickle_file

if __name__ == '__main__':
    feature_args=read_pickle_file('pipeline_profiler_args.bin')
    # print(feature_args)
    args = parse_args()
    # compute_features(args)
    # run_pipeline(args)
    # process_feature_performance(args)
