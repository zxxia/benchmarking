"""Interface Abstraction."""


class Temporal(object):
    """Temporal Pruning Interface Abstraction."""

    def __init__(self, config):
        # TODO load the configs
        self.config = config
        raise NotImplementedError

    def run(self, segment, decision, results):
        raise NotImplementedError


class Spatial(object):
    """Spatial Pruning Interface Abstraction."""

    def __init__(self, config):
        # TODO load the configs
        raise NotImplementedError

    def run(self, segment, decision, results):
        raise NotImplementedError


class Model(object):
    """Model Pruning Interface Abstraction."""

    def __init__(self, config):
        # TODO load the configs
        raise NotImplementedError

    def run(self, segment, decision, results):
        raise NotImplementedError


class Decision(object):
    """Pruning decision on a single frame."""

    def __init__(self, skip=False, resolution=None, cropping=None, qp=2,
                 dnn='resnet101', tracker=None):
        assert isinstance(skip, bool), "skip needs to be bool."
        assert resolution is None or \
            (isinstance(resolution, tuple) and len(resolution) == 2 and
             all(isinstance(val, int) for val in resolution)), \
            "resolution needs to be either None or a tuple of 2 integers."
        assert cropping is None or isinstance(cropping, list) and \
            all(isinstance(box, list) and len(box) == 4 for box in cropping), \
            "cropping needs to be None or a list of length 4 lists."
        assert isinstance(qp, int) and 1 <= qp <= 51, \
            "qp needs to be an integer."
        assert dnn is None or isinstance(
            dnn, str), "dnn needs to be None or a string."
        assert tracker is None or isinstance(tracker, str), \
            "tracker needs to be None or a string."

        self.skip = skip
        self.resolution = resolution
        self.cropping = cropping
        self.qp = qp
        self.dnn = dnn
        self.tracker = tracker

    def __str__(self):
        """Return the content of Decision in a dictionary."""
        return f"skip: {self.skip}, resolution: {self.resolution}, cropping: {self.cropping}, qp: {self.qp}, dnn: {self.dnn}, tracker: {self.tracker}"
