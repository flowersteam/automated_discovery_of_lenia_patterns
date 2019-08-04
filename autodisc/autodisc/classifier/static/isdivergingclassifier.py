import autodisc as ad
import numpy as np


class IsDivergingClassifier(ad.core.Classifier):

    @staticmethod
    def default_config():
        default_config = ad.core.Classifier.default_config()

        default_config.statistic_name = 'activation_mass'
        default_config.phase1_timepoints = (1/3, 2/3)
        default_config.phase2_timepoints = (2/3, 1)
        default_config.max_average_diff = 0.05              # in percent

        return default_config


    def calc(self, observations, statistics=None):

        if statistics is None:
            raise ValueError('IsDivergingClassifier needs statistics!')

        if self.config.statistic_name not in statistics:
            raise ValueError('Can not find {!r} in statistics!'.format(self.config.statistic_name))

        data = statistics[self.config.statistic_name]

        phase1_start_idx = int(len(data) * self.config.phase1_timepoints[0])
        phase1_end_idx = int(len(data) * self.config.phase1_timepoints[1])

        phase2_start_idx = int(len(data) * self.config.phase2_timepoints[0])
        phase2_end_idx = int(len(data) * self.config.phase2_timepoints[1])

        phase1_avg = np.nanmean(data[phase1_start_idx:phase1_end_idx])
        phase2_avg = np.nanmean(data[phase2_start_idx:phase2_end_idx])

        ratio = phase2_avg / phase1_avg

        if ratio > 1 + self.config.max_average_diff:
            category = 1
        elif ratio < 1 - self.config.max_average_diff:
            category = -1
        else:
            category = 0

        return category