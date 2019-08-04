import autodisc as ad
import numpy as np

class ObservationDifferenceStatistic(ad.core.SystemStatistic):
    def __init__(self):

        super().__init__()

        self.data['max_absolute_observation_difference'] = []
        self.data['max_absolute_observation_difference_mean'] = []
        self.data['max_absolute_observation_difference_std'] = []

        self.data['max_positive_observation_difference'] = []
        self.data['max_positive_observation_difference_mean'] = []
        self.data['max_positive_observation_difference_std'] = []

        self.data['max_negative_observation_difference'] = []
        self.data['max_negative_observation_difference_mean'] = []
        self.data['max_negative_observation_difference_std'] = []


    def reset(self):
        # set all statistics to zero
        for key in self.data.keys():
            self.data[key] = []


    def calc_after_run(self, system, all_obs):
        '''Calculates the final statistics for lenia observations after a run is completed'''

        max_positive_observation_difference = np.zeros(len(all_obs)) * np.nan
        max_negative_observation_difference = np.zeros(len(all_obs)) * np.nan
        max_absolute_observation_difference = np.zeros(len(all_obs)) * np.nan

        for obs_idx in range(1, len(all_obs)):
            diff = all_obs[obs_idx] - all_obs[obs_idx-1]

            max_diff = np.nanmax(diff)
            min_diff = np.nanmin(diff)

            max_positive_observation_difference[obs_idx] = max_diff if max_diff > 0 else 0
            max_negative_observation_difference[obs_idx] = -min_diff if min_diff < 0 else 0
            max_absolute_observation_difference[obs_idx] = np.nanmax([max_positive_observation_difference[obs_idx], max_negative_observation_difference[obs_idx]])

        self.data['max_absolute_observation_difference'] = max_absolute_observation_difference
        self.data['max_absolute_observation_difference_mean'] = np.nanmean(max_absolute_observation_difference)
        self.data['max_absolute_observation_difference_std'] = np.nanstd(max_absolute_observation_difference)

        self.data['max_positive_observation_difference'] = max_positive_observation_difference
        self.data['max_positive_observation_difference_mean'] = np.nanmean(max_positive_observation_difference)
        self.data['max_positive_observation_difference_std'] = np.nanstd(max_positive_observation_difference)

        self.data['max_negative_observation_difference'] = max_negative_observation_difference
        self.data['max_negative_observation_difference_mean'] = np.nanmean(max_negative_observation_difference)
        self.data['max_negative_observation_difference_std'] = np.nanstd(max_negative_observation_difference)