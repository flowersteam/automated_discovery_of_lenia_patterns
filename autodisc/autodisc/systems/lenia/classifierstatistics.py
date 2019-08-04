import autodisc as ad


class LeniaClassifierStatistics(ad.core.SystemStatistic):

    def __init__(self):

        super().__init__()

        self.data['classifier_animal'] = []
        self.data['classifier_diverging'] = []
        self.data['classifier_stable_fixpoint_solution'] = []
        self.data['classifier_moving'] = []

        self.is_lenia_animal_classifier = ad.systems.lenia.IsLeniaAnimalClassifier()
        self.is_diverging_classifier = ad.classifier.static.IsDivergingClassifier()
        self.stable_fix_point_solution_classifier = ad.classifier.static.StableFixPointSolutionClassifier()
        self.is_moving_classifier = ad.classifier.static.IsMovingClassifier()


    def reset(self):
        # set all statistics to zero
        for key in self.data.keys():
            self.data[key] = []


    def calc_after_run(self, system, all_obs):
        '''Calculates the final statistics for lenia observations after a run is completed'''

        # collect statistics from system which might be needed for the classifiers
        statistics_data = dict()
        for stat in system.statistics:
            statistics_data = {**stat.data, **statistics_data}

        self.is_lenia_animal_classifier.config.r = system.run_parameters['R']
        self.data['classifier_animal'] = self.is_lenia_animal_classifier.calc(all_obs, statistics_data)

        self.data['classifier_diverging'] = self.is_diverging_classifier.calc(all_obs, statistics_data)

        self.data['classifier_stable_fixpoint_solution'] = self.stable_fix_point_solution_classifier.calc(all_obs, statistics_data)

        self.data['classifier_moving'] = self.is_moving_classifier.calc(all_obs, statistics_data)