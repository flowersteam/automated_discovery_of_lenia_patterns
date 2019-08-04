import exputils
import autodisc as ad
import numpy as np
import os
import importlib.util


def calc_error_in_goalspace_between_goal_usedpolicy(repetition_data):
    data = []

    for rep_id, rep_explorer in repetition_data.items():
        cur_differences = rep_explorer.goal_space_representation.calc_distance(np.array(rep_explorer.statistics['target_goals']), np.array(rep_explorer.statistics['reached_goals']))

        data = exputils.misc.numpy_vstack_2d_default(data, cur_differences)

    if len(np.shape(data)) == 1:
        data = np.array([data])
    else:
        data = np.array(data)

    statistic = dict()
    statistic['data'] = data
    statistic['mean'] = np.nanmean(data, axis=0)
    statistic['std'] = np.nanstd(data, axis=0)

    return statistic


def calc_error_in_goalspace_between_goal_bestpolicy(repetition_data):

    data = []

    for rep_id, rep_explorer in repetition_data.items():

        cur_differences = []

        for run_data in rep_explorer.data:
            if run_data.target_goal is not None and run_data.source_policy_idx is not None:
                dist = rep_explorer.goal_space_representation.calc_distance(run_data.target_goal, rep_explorer.data[run_data.source_policy_idx].reached_goal)
                cur_differences.append(dist)

        data = exputils.misc.numpy_vstack_2d_default(data, cur_differences)

    if len(np.shape(data)) == 1:
        data = np.array([data])
    else:
        data = np.array(data)

    statistic = dict()
    statistic['data'] = data
    statistic['mean'] = np.nanmean(data, axis=0)
    statistic['std'] = np.nanstd(data, axis=0)

    return statistic


def calc_classifier_is_dead(repetition_data):

    # collect all classifications
    data = []
    ratio = []
    for rep_id, rep_explorer in repetition_data.items():

        cur_rep_data = []
        for run_data in rep_explorer.data:
            cur_rep_data.append(run_data.statistics['is_dead'])

        ratio.append(np.sum(cur_rep_data)/len(cur_rep_data))
        data = exputils.misc.numpy_vstack_2d_default(data, cur_rep_data)

    if len(np.shape(data)) == 1:
        data = np.array([data])
    else:
        data = np.array(data)

    # compute the
    statistic = dict()
    statistic['data'] = data
    statistic['ratio'] = ratio
    statistic['mean'] = np.nanmean(ratio)
    statistic['std'] = np.nanstd(ratio)

    return statistic


def calc_classifier_animal(repetition_data):

    # collect all classifications
    data = []
    ratio = []
    for rep_id, rep_explorer in repetition_data.items():

        cur_rep_data = []
        for run_data in rep_explorer.data:
            cur_rep_data.append(run_data.statistics['classifier_animal'])

        ratio.append(np.sum(cur_rep_data)/len(cur_rep_data))
        data = exputils.misc.numpy_vstack_2d_default(data, cur_rep_data)

    if len(np.shape(data)) == 1:
        data = np.array([data])
    else:
        data = np.array(data)

    # compute the
    statistic = dict()
    statistic['data'] = data
    statistic['ratio'] = ratio
    statistic['mean'] = np.nanmean(ratio)
    statistic['std'] = np.nanstd(ratio)

    return statistic


def calc_classifier_diverging(repetition_data):

    # collect all classifications
    data = []

    ratio_negative = []
    ratio_not = []
    ratio_positive = []

    for rep_id, rep_explorer in repetition_data.items():

        cur_rep_data = []
        for run_data in rep_explorer.data:
            cur_rep_data.append(run_data.statistics['classifier_diverging'])

        ratio_negative.append(np.sum(cur_rep_data == -1) / np.sum(~np.isnan(cur_rep_data)))
        ratio_not.append(np.sum(cur_rep_data == -1) / np.sum(~np.isnan(cur_rep_data)))
        ratio_positive.append(np.sum(cur_rep_data == 1) / np.sum(~np.isnan(cur_rep_data)))

        data = exputils.misc.numpy_vstack_2d_default(data, cur_rep_data)

    if len(np.shape(data)) == 1:
        data = np.array([data])
    else:
        data = np.array(data)

    # compute the
    statistic = dict()
    statistic['data'] = data

    statistic['ratio_negative'] = ratio_negative
    statistic['mean_negative'] = np.nanmean(ratio_negative)
    statistic['std_negative'] = np.nanstd(ratio_negative)

    statistic['ratio_not'] = ratio_not
    statistic['mean_not'] = np.nanmean(ratio_not)
    statistic['std_not'] = np.nanstd(ratio_not)

    statistic['ratio_positive'] = ratio_positive
    statistic['mean_positive'] = np.nanmean(ratio_positive)
    statistic['std_positive'] = np.nanstd(ratio_positive)

    return statistic


def calc_classifier_stable_fixpoint_solution(repetition_data):

    # collect all classifications
    data = []
    ratio = []
    for rep_id, rep_explorer in repetition_data.items():

        cur_rep_data = []
        for run_data in rep_explorer.data:
            cur_rep_data.append(run_data.statistics['classifier_stable_fixpoint_solution'])

        ratio.append(np.sum(cur_rep_data)/len(cur_rep_data))
        data = exputils.misc.numpy_vstack_2d_default(data, cur_rep_data)

    if len(np.shape(data)) == 1:
        data = np.array([data])
    else:
        data = np.array(data)

    # compute the
    statistic = dict()
    statistic['data'] = data
    statistic['ratio'] = ratio
    statistic['mean'] = np.nanmean(ratio)
    statistic['std'] = np.nanstd(ratio)

    return statistic


def calc_classifier_moving(repetition_data):

    # collect all classifications
    data = []
    ratio = []
    for rep_id, rep_explorer in repetition_data.items():

        cur_rep_data = []
        for run_data in rep_explorer.data:
            cur_rep_data.append(run_data.statistics['classifier_moving'])

        ratio.append(np.sum(cur_rep_data)/len(cur_rep_data))
        data = exputils.misc.numpy_vstack_2d_default(data, cur_rep_data)

    if len(np.shape(data)) == 1:
        data = np.array([data])
    else:
        data = np.array(data)

    # compute the
    statistic = dict()
    statistic['data'] = data
    statistic['ratio'] = ratio
    statistic['mean'] = np.nanmean(ratio)
    statistic['std'] = np.nanstd(ratio)

    return statistic


def collect_run_parameters(repetition_data):
    '''Save some of the run paramters per run in an easier format to read.'''
    num_of_b_components = 4

    t_parameters = []
    r_parameters = []
    m_parameters = []
    s_parameters = []

    b_parameters = dict()
    for b_idx in range(num_of_b_components):
        b_parameters[b_idx] = []

    for rep_id, rep_explorer in repetition_data.items():

        cur_t_parameters = []
        cur_r_parameters = []
        cur_m_parameters = []
        cur_s_parameters = []

        cur_b_parameters = dict()
        for b_idx in range(num_of_b_components):
            cur_b_parameters[b_idx] = []

        for run_data in rep_explorer.data:

            cur_t_parameters.append(run_data.run_parameters['T'])
            cur_r_parameters.append(run_data.run_parameters['R'])
            cur_m_parameters.append(run_data.run_parameters['m'])
            cur_s_parameters.append(run_data.run_parameters['s'])

            for b_idx in range(num_of_b_components):
                if len(run_data.run_parameters['b']) > b_idx:
                    cur_b_parameters[b_idx].append(run_data.run_parameters['b'][b_idx])
                else:
                    cur_b_parameters[b_idx].append(0)

        t_parameters = exputils.misc.numpy_vstack_2d_default(t_parameters, cur_t_parameters)
        r_parameters = exputils.misc.numpy_vstack_2d_default(r_parameters, cur_r_parameters)
        m_parameters = exputils.misc.numpy_vstack_2d_default(m_parameters, cur_m_parameters)
        s_parameters = exputils.misc.numpy_vstack_2d_default(s_parameters, cur_s_parameters)

        for b_idx in range(num_of_b_components):
            b_parameters[b_idx] = exputils.misc.numpy_vstack_2d_default(b_parameters[b_idx], cur_b_parameters[b_idx])

    stat = dict()

    if len(np.shape(t_parameters)) == 1:
        t_parameters = np.array([t_parameters])
    else:
        t_parameters = np.array(t_parameters)

    if len(np.shape(r_parameters)) == 1:
        r_parameters = np.array([r_parameters])
    else:
        r_parameters = np.array(r_parameters)

    if len(np.shape(m_parameters)) == 1:
        m_parameters = np.array([m_parameters])
    else:
        m_parameters = np.array(m_parameters)

    if len(np.shape(s_parameters)) == 1:
        s_parameters = np.array([s_parameters])
    else:
        s_parameters = np.array(s_parameters)

    for b_idx in range(num_of_b_components):
        if len(np.shape(b_parameters[b_idx])) == 1:
            b_parameters[b_idx] = np.array([b_parameters[b_idx]])
        else:
            b_parameters[b_idx] = np.array(b_parameters[b_idx])

    stat['T'] = t_parameters
    stat['R'] = r_parameters
    stat['m'] = m_parameters
    stat['s'] = s_parameters
    stat['b'] = b_parameters

    return stat


def load_data(repetition_directories):

    data = dict()

    for repetition_directory in sorted(repetition_directories):

        # get id of the repetition from its foldername
        numbers_in_string = [int(s) for s in os.path.basename(repetition_directory).split('_') if s.isdigit()]
        repetition_id = numbers_in_string[0]

        # load the full explorer without observations and add its config
        explorer = ad.explorers.GoalSpaceExplorer.load_explorer(os.path.join(repetition_directory, 'results'), run_ids=[], load_observations=False, verbose=False)
        explorer.data.config.load_observations = True
        explorer.data.config.memory_size_observations = 1

        spec = importlib.util.spec_from_file_location('experiment_config', os.path.join(repetition_directory, 'experiment_config.py'))
        experiment_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment_config_module)
        explorer.config = experiment_config_module.get_explorer_config()

        data[repetition_id] = explorer

    return data


if __name__ == '__main__':

    experiments = '.'
    #experiments = './experiments/experiment_000001/'

    statistics = [('error_in_goalspace_between_goal_bestpolicy', calc_error_in_goalspace_between_goal_bestpolicy),
                  ('error_in_goalspace_between_goal_usedpolicy', calc_error_in_goalspace_between_goal_usedpolicy),
                  ('classifier_dead', calc_classifier_is_dead),
                  ('classifier_animal', calc_classifier_animal),
                  ('classifier_diverging', calc_classifier_diverging),
                  ('classifier_stable_fixpoint_solution', calc_classifier_stable_fixpoint_solution),
                  ('classifier_moving', calc_classifier_moving),
                  ('run_parameters', collect_run_parameters)
                  ]

    exputils.calc_statistics_over_repetitions(statistics, load_data, experiments, recalculate_statistics=False, verbose=True)
