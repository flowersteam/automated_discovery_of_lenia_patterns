import autodisc as ad
import numpy as np

class IsLeniaAnimalClassifierV2(ad.core.Classifier):

    @staticmethod
    def default_config():
        default_config = ad.core.Classifier.default_config()
        default_config.r = 1
        default_config.tol = 0.1
        default_config.obs1_idx = -1
        default_config.obs2_idx = -2
        default_config.min_activity = 0.8

        return default_config


    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

        self.segmented_image = None
        self.finite_segments = None


    def calc(self, observations, statistics=None):

        self.finite_segments_obs1 = None
        self.segmented_image_obs1 = None
        self.finite_segments_obs2 = None
        self.segmented_image_obs2 = None
        self.animal_segments_obs1 = []
        self.animal_segments_obs2 = []

        obs1 = observations[self.config.obs1_idx]

        self.finite_segments_obs1, (self.segmented_image_obs1, _) = ad.helper.statistics.calc_is_segments_finite_v2(obs1, tol=self.config.tol, r=self.config.r)

        if not self.finite_segments_obs1:
            return False

        else:

            activity = np.sum(obs1)

            for seg_idx in self.finite_segments_obs1:
                cur_activity = np.sum(obs1[self.segmented_image_obs1 == seg_idx])

                if cur_activity / activity >= self.config.min_activity:
                    self.animal_segments_obs1.append(seg_idx)

            if not self.animal_segments_obs1:
                return False

            else:
                obs2 = observations[self.config.obs2_idx]

                self.finite_segments_obs2, (self.segmented_image_obs2, _) = ad.helper.statistics.calc_is_segments_finite_v2(obs1, tol=self.config.tol, r=self.config.r)

                activity = np.sum(obs2)

                for seg_idx in self.finite_segments_obs2:
                    cur_activity = np.sum(obs2[self.segmented_image_obs2 == seg_idx])

                    if cur_activity / activity >= self.config.min_activity:
                        self.animal_segments_obs2.append(seg_idx)

                if not self.animal_segments_obs2:
                    return False

        return True