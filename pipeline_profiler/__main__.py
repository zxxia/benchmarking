import sys  
sys.path.append('./') 
# print('\n'.join(sys.path))
"""Profiler module main function. 
This module will first compute_features and run the pipeline on the given video
the feature computed and the performance csv will be output for user to double check.
Then this module will process the feature an performance and build a correspondence map (json format) """
from pipeline_profiler.parser import parse_args
from pipeline_profiler.run_pipeline import run_pipeline
from pipeline_profiler.process_feature_performance import process_feature_performance
from feature_scanner.compute_features import *

if __name__ == '__main__':
    args = parse_args()
    # compute_features(args)
    # run_pipeline(args)
    process_feature_performance(args)
