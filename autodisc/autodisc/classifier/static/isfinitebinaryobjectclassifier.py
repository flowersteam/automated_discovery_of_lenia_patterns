import autodisc as ad

class IsFiniteBinaryObjectClassifier(ad.core.Classifier):

    @staticmethod
    def default_config():
        default_config = ad.core.Classifier.default_config()
        default_config.r = 1
        default_config.tol = 0.1

        return default_config


    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

        self.segmented_image = None
        self.finite_segments = None


    def calc(self, observations, statistics=None):

        self.finite_segments, (self.segmented_image, _), _ = ad.helper.statistics.calc_is_segments_finite(observations, tol=self.config.tol, r=self.config.r)

        if self.finite_segments:
            return True
        else:
            return False


