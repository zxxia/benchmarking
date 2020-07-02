"""Abstraction of a pipeline's profile. """


class Profile(object):
    def __init__(self, root):

        # TODO: read temporal profile in root
        # TODO: read spatial profile in root
        # TODO: read model profile in root
        # TODO: need to contain temporal, spatial, and model pruning profiles
        raise NotImplementedError("Profile base class not implemented.")

    def temporal_profile(self):
        """Return temporal profile."""
        raise NotImplementedError

    def spatial_profile(self):
        """Return spatial profile."""
        raise NotImplementedError

    def model_profile(self):
        """Return model profile."""
        raise NotImplementedError
