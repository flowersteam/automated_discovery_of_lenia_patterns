import autodisc as ad
import numpy as np

class IsMovingClassifier(ad.core.Classifier):

    @staticmethod
    def default_config():
        default_config = ad.core.Classifier.default_config()
        default_config.min_distance_from_mean_pos = 10
        default_config.last_n_steps = 50

        return default_config


    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

        self.segmented_image = None
        self.finite_segments = None


    def calc(self, observations, statistics=None):

        # see if there is a change of the center position in the last x time steps

        # calculate the relative position of the activation center beginning fom the last n step
        # we are not using the activation_center statistic, because it changes drastically if the center is near the image border

        # get shifts of each step
        movement_angles_in_rad = np.radians(statistics['activation_center_movement_angle'][-self.config.last_n_steps:])
        movement_distance = statistics['activation_center_velocity'][-self.config.last_n_steps:]
        x_shift = np.cos(movement_angles_in_rad) * movement_distance
        y_shift = np.sin(movement_angles_in_rad) * movement_distance
        x_shift[movement_distance == 0] = 0
        y_shift[movement_distance == 0] = 0

        # get relative positions
        relative_x_pos = np.cumsum(x_shift)
        relative_y_pos = np.cumsum(y_shift)
        relative_pos = np.vstack((relative_x_pos, relative_y_pos)).transpose()

        # relative mean position in last n steps
        mean_pos = np.nanmean(relative_pos, axis=0)

        # calc dist of each position to the mean position
        dist_to_mean_pos = np.linalg.norm(mean_pos - relative_pos, axis=1)

        # detect movement if the distance of one pos is larger than x to the mean pos
        return np.nanmax(dist_to_mean_pos) >= self.config.min_distance_from_mean_pos