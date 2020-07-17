import sys  
sys.path.append('./') 
# print('\n'.join(sys.path))
from feature_scanner.parser import parse_args
from feature_scanner.compute_features import compute_features
args = parse_args()
compute_features(args)
