from interface import Model, Spatial, Temporal


class Pipeline(object):
    """Pipeline base class."""

    def __init__(self, temporal_prune: Temporal, spatial_prune: Spatial,
                 model_prune: Model, configurations):
        """Pipeline base class init function."""
        self.temporal_prune = temporal_prune
        self.spatial_prune = spatial_prune
        self.model_prune_tracking = model_prune
        # TODO: need to load configurations

        raise NotImplementedError("Pipeline base class not implemented.")

    def run(self, video):
        # TODO: consume videos and apply the pruning strategies
        raise NotImplementedError("Pipeline base class not implemented.")
