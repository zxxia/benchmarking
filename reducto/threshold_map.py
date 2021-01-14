from typing import List, Mapping, Tuple

import numpy as np
from sklearn import neighbors


class ThresholdMap:
    """Reductor Threshold map.

    Args
    """

    def __init__(self, knn, diff_range: Tuple[float, float],
                 thresh_candidate: List[float], feat_dim: int, feat_type: str):
        # NOTE n_neighbors must be greater than number of segment
        self.knn = knn
        self.diff_range = diff_range
        self.thresh_candidate = thresh_candidate
        self.feature_dim = feat_dim
        self.feat_type = feat_type

    def get_thresh(self, diff_vector: List[float]):  # , motion_vector=None):
        """Use frame differences to find a suitable thrsehold."""
        diff_vector = np.array(ThresholdMap._histogram(
            diff_vector, self.diff_range, self.feature_dim))[np.newaxis, :]
        pred_thresh = self.knn.predict(diff_vector).item()
        distance, _ = self.knn.kneighbors(diff_vector, return_distance=True)
        return self.thresh_candidate[pred_thresh], distance

    @staticmethod
    def build(eval_results: List[Mapping[float, float]],
              diff_vectors: List[List[float]], feat_type: str,
              knn_neighbors: int = 5, feature_dim: int = 30,
              target_f1: float = 0.9):
        """Build a threshold map.

        Args
            eval_results: a list of mappings from thresholds to F1 scores. Each
                mapping represents the evaluation result of a segment.
            diff_vectors: a list of frame differences of a segment.
        """
        assert len(eval_results) == len(diff_vectors)
        # diff_value_range = (min_diff_value, max_diff_value)
        diff_value_range = None
        for diff_vec in diff_vectors:
            if diff_value_range is None:
                diff_value_range = (min(diff_vec), max(diff_vec))
            else:
                diff_value_range = (min([min(diff_vec), diff_value_range[0]]),
                                    max([max(diff_vec), diff_value_range[1]]))
        assert diff_value_range is not None

        # optimal_thresh: [(distri_vector, optimal_thresh)], each segment has a
        # tuple
        optimal_thresh = []
        thresh_candidate = sorted(eval_results[0].keys())
        # dp_er Mapping[DiffProcessor, Mapping[float, Dict]]
        for eval_result, diff_vec in zip(eval_results, diff_vectors):
            optimal_thresh.append((
                ThresholdMap._histogram(
                    diff_vec, diff_value_range, feature_dim),
                ThresholdMap._get_optimal_thresh(eval_result, target_f1)
            ))

        knn = neighbors.KNeighborsClassifier(
            n_neighbors=knn_neighbors, weights='distance')
        x = np.array([hist for hist, _ in optimal_thresh])
        _y = [(thresh_candidate.index(opt) if opt in thresh_candidate else 0)
              for _, opt in optimal_thresh]
        y = np.array(_y)
        knn.fit(x, y)
        # hash_table = {
        #     'knn': knn,
        #     'diff range': diff_value_range,
        #     'dim': feature_dim,
        #     'threhold candidate': thresh_candidate
        # }
        return ThresholdMap(knn, diff_value_range, thresh_candidate,
                            feature_dim, feat_type)

    @staticmethod
    def _histogram(diff_vector, dist_range, feature_dim):
        hist, _ = np.histogram(diff_vector, bins=feature_dim, range=dist_range)
        return hist / len(diff_vector)

    @staticmethod
    def _get_optimal_thresh(er, target_acc):
        optimal_thresh = 0.0
        for thresh, result_accs in er.items():
            thresh = float(thresh)
            # result_cross_query = min([abs(x) for x in result.values()])
            if min(result_accs) > target_acc and thresh > optimal_thresh:
                optimal_thresh = thresh
        return optimal_thresh
