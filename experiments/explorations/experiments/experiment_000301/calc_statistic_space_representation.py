import exputils
import autodisc as ad
import numpy as np
import os

def calc_statistic_space_representation(repetition_data):

    # load representation
    data = []

    config = ad.representations.static.PytorchNNRepresentation.default_config()
    config.initialization.type = 'load_pretrained_model'
    config.initialization.load_from_model_path = '../../../post_train_analytic_behavior_space/training/training/models/best_weight_model.pth'
    representation_model = ad.representations.static.PytorchNNRepresentation(config)

    for rep_id, rep_data in repetition_data.items():

        cur_rep_data = []
        for run_data in rep_data:
            cur_representation = representation_model.calc(run_data.observations, run_data.statistics)
            cur_rep_data.append(cur_representation)

        data.append(cur_rep_data)

    if len(np.shape(data)) == 1:
        data = np.array([data])
    else:
        data = np.array(data)

    # compute the
    statistic = dict()
    statistic['data'] = data

    return statistic


def calc_parameter_initstate_space_representation(repetition_data):

    # load representation
    data = []

    config = ad.representations.static.PytorchNNRepresentation.default_config()
    config.initialization.type = 'load_pretrained_model'
    config.initialization.load_from_model_path = '../../../post_train_analytic_parameter_space/training/training/models/best_weight_model.pth'
    representation_model = ad.representations.static.PytorchNNRepresentation(config)

    for rep_id, rep_data in repetition_data.items():

        cur_rep_data = []
        for run_data in rep_data:

            # the representation will only use the final observation, thus repack the observations to give it the first, i.e. the CPPN generated image
            new_obs = dict(states=[run_data.observations['states'][0]],
                           timepoints=[0])

            cur_representation = representation_model.calc(new_obs)
            cur_rep_data.append(cur_representation)

        data.append(cur_rep_data)

    if len(np.shape(data)) == 1:
        data = np.array([data])
    else:
        data = np.array(data)

    # compute the
    statistic = dict()
    statistic['data'] = data

    return statistic


def load_data(repetition_directories):

    data = dict()

    for repetition_directory in sorted(repetition_directories):

        # get id of the repetition from its foldername
        numbers_in_string = [int(s) for s in os.path.basename(repetition_directory).split('_') if s.isdigit()]
        repetition_id = numbers_in_string[0]

        # load the full explorer without observations and add its config
        datahandler_config = ad.ExplorationDataHandler.default_config()
        datahandler_config.memory_size_observations = 1

        rep_data = ad.ExplorationDataHandler.create(config=datahandler_config, directory=os.path.join(repetition_directory, 'results'))
        rep_data.load(load_observations=False, verbose=True)

        data[repetition_id] = rep_data

    return data


if __name__ == '__main__':

    experiments = '.'

    statistics = [('statistic_space_representation', calc_statistic_space_representation),
                  ('parameter_initstate_space_representation', calc_parameter_initstate_space_representation)
                  ]

    exputils.calc_statistics_over_repetitions(statistics, load_data, experiments, recalculate_statistics=False, verbose=True)
