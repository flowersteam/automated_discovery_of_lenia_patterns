import autodisc as ad
import numpy as np


class StatisticRepresentation(ad.core.Representation):

    @staticmethod
    def default_config():
        default_config = ad.core.Representation.default_config()
        default_config.statistics = []
        default_config.distance_function = None

        return default_config


    def calc(self, observations, statistics):

        representation = np.zeros(len(self.config.statistics))
        for idx, stat_name in enumerate(self.config.statistics):
            representation[idx] = ad.helper.misc.get_sub_dictionary_variable(statistics, stat_name)

        return representation


    def calc_distance(self, representation1, representation2):
        if self.config.distance_function is None:
            return super().calc_distance(representation1, representation2)
        else:
            return self.config.distance_function(representation1, representation2, self.config)

