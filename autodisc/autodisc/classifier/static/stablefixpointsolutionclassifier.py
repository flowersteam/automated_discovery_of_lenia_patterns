import autodisc as ad
import numpy as np

class StableFixPointSolutionClassifier(ad.core.Classifier):

    @staticmethod
    def default_config():
        default_config = ad.core.Classifier.default_config()

        default_config.min_difference_tolerance = 0.1
        default_config.n_steps = 2

        return default_config


    def calc(self, observations, statistics=None):

        obs_max_diffs = self.calc_obs_max_diffs(observations)

        if np.nanmax(obs_max_diffs) > self.config.min_difference_tolerance:
            return False
        else:
            return True


    def calc_obs_max_diffs(self, observations):
        '''The maximum absolute difference in between the <config.n_steps> last observations.'''

        max_diff = []

        for obs_idx in range((len(observations) - self.config.n_steps), len(observations)):
            diff = observations[obs_idx] - observations[obs_idx-1]
            max_diff.append(np.nanmax(np.abs(diff)))

        return max_diff