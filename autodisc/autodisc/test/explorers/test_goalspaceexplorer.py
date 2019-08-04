import autodisc as ad
import numpy as np
import os

def test_statistic_goalspace():
    # use lenia as test system

    system = ad.systems.Lenia()

    config = ad.explorers.GoalSpaceExplorer.default_config()
    config.seed = 1
    config.num_of_random_initialization = 2

    # Parameter 2: R
    parameter = ad.Config()
    parameter.name = 'R'
    parameter.type = 'sampling'
    parameter.init = ('discrete', 2, 20)
    parameter.mutate = {'type': 'discrete', 'distribution': 'gauss', 'sigma': 0.5, 'min': 2, 'max': 20}
    config.run_parameters.append(parameter)

    # Parameter 3: T
    parameter = ad.Config()
    parameter.name = 'T'
    parameter.type = 'sampling'
    parameter.init = ('discrete', 1, 20)
    parameter.mutate = {'type': 'discrete', 'distribution': 'gauss', 'sigma': 0.5, 'min': 1, 'max': 20}
    config.run_parameters.append(parameter)

    # which statistics are used as a goal space
    config.goal_space_representation.type = 'statistics'
    config.goal_space_representation.config = ad.representations.static.StatisticRepresentation.default_config()
    config.goal_space_representation.config.statistics = ['activation_mass_mean',
                                                          'activation_mass_std']

    # how are goals sampled
    config.goal_selection.type = 'random'
    config.goal_selection.sampling = [(0, 1),
                                      (0, 0.4)]

    explorer = ad.explorers.GoalSpaceExplorer(system=system, config=config)
    explorer.run(5, verbose=False)

    assert len(explorer.data) == 5
    assert np.shape(explorer.statistics.reached_initial_goals) == (2, 2)  # no statistics should have been computed
    assert np.shape(explorer.statistics.target_goals) == (3, 2)    # no statistics should have been computed
    assert np.shape(explorer.statistics.reached_goals) == (3, 2)    # no statistics should have been computed


def goal_space_func_with_config(observations, statistics, config):

    # check if given parameters are correct
    assert config == 'test'
    assert 'states' in observations
    assert 'timepoints' in observations
    assert 'activation_mass' in statistics

    return [1, 2, 3]


def goal_space_func_without_config(observations, statistics):
    # check if given parameters are correct
    assert 'states' in observations
    assert 'timepoints' in observations
    assert 'activation_mass' in statistics

    return [10, 20, 30]


def test_function_goalspace():
    # use lenia as test system

    system = ad.systems.Lenia()

    config = ad.explorers.GoalSpaceExplorer.default_config()
    config.seed = 1
    config.num_of_random_initialization = 2

    # Parameter 2: R
    parameter = ad.Config()
    parameter.name = 'R'
    parameter.type = 'sampling'
    parameter.init = ('discrete', 2, 20)
    parameter.mutate = {'type': 'discrete', 'distribution': 'gauss', 'sigma': 0.5, 'min': 2, 'max': 20}
    config.run_parameters.append(parameter)

    # Parameter 3: T
    parameter = ad.Config()
    parameter.name = 'T'
    parameter.type = 'sampling'
    parameter.init = ('discrete', 1, 20)
    parameter.mutate = {'type': 'discrete', 'distribution': 'gauss', 'sigma': 0.5, 'min': 1, 'max': 20}
    config.run_parameters.append(parameter)

    # which statistics are used as a goal space
    config.goal_space_representation.type = 'function'
    config.goal_space_representation.config = ad.representations.FunctionRepresentation.default_config()
    config.goal_space_representation.config.function = goal_space_func_without_config

    # how are goals sampled
    config.goal_selection.type = 'random'
    config.goal_selection.sampling = [(0, 1),
                                      (0, 0.4),
                                      (0, 0.5)]

    explorer = ad.explorers.GoalSpaceExplorer(system=system, config=config)
    explorer.run(5, verbose=False)

    assert len(explorer.data) == 5
    assert np.shape(explorer.statistics.reached_initial_goals) == (2, 3)  # no statistics should have been computed
    assert np.shape(explorer.statistics.target_goals) == (3, 3)    # no statistics should have been computed
    assert np.shape(explorer.statistics.reached_goals) == (3, 3)    # no statistics should have been computed

    assert np.all(explorer.statistics.reached_goals == [[10, 20, 30], [10, 20, 30], [10, 20, 30]])

    del explorer

    #########################################################
    # with config

    config.goal_space_representation.config.function = goal_space_func_with_config
    config.goal_space_representation.config.config = 'test'

    explorer = ad.explorers.GoalSpaceExplorer(system=system, config=config)
    explorer.run(5, verbose=False)

    assert len(explorer.data) == 5
    assert np.shape(explorer.statistics.reached_initial_goals) == (2, 3)  # no statistics should have been computed
    assert np.shape(explorer.statistics.target_goals) == (3, 3)  # no statistics should have been computed
    assert np.shape(explorer.statistics.reached_goals) == (3, 3)  # no statistics should have been computed

    assert np.all(explorer.statistics.reached_goals == [[1, 2, 3], [1, 2, 3], [1, 2, 3]])

    del explorer

    #########################################################
    # function as string

    config.goal_space_representation.config.function = 'autodisc.test.explorers.test_goalspaceexplorer.goal_space_func_with_config'
    config.goal_space_representation.config.config = 'test'

    explorer = ad.explorers.GoalSpaceExplorer(system=system, config=config)
    explorer.run(5, verbose=False)

    assert len(explorer.data) == 5
    assert np.shape(explorer.statistics.reached_initial_goals) == (2, 3)  # no statistics should have been computed
    assert np.shape(explorer.statistics.target_goals) == (3, 3)  # no statistics should have been computed
    assert np.shape(explorer.statistics.reached_goals) == (3, 3)  # no statistics should have been computed

    assert np.all(explorer.statistics.reached_goals == [[1, 2, 3], [1, 2, 3], [1, 2, 3]])

    del explorer


def test_cppn_evolution():
    # use lenia as test system

    system = ad.systems.Lenia()

    config = ad.explorers.GoalSpaceExplorer.default_config()
    config.seed = 1
    config.num_of_random_initialization = 2

    parameter = ad.Config()
    parameter.name = 'init_state'
    parameter.type = 'cppn_evolution'

    parameter.init = ad.cppn.TwoDMatrixCCPNNEATEvolution.default_config()
    parameter.init.neat_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_neat_single.cfg')
    parameter.init.n_generations = 1
    parameter.init.best_genome_of_last_generation = True

    parameter.mutate = ad.cppn.TwoDMatrixCCPNNEATEvolution.default_config()
    parameter.mutate.neat_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_neat_single.cfg')
    parameter.mutate.n_generations = 2
    parameter.mutate.best_genome_of_last_generation = True

    config.run_parameters.append(parameter)

    # Parameter 2: R
    parameter = ad.Config()
    parameter.name = 'R'
    parameter.type = 'sampling'
    parameter.init = ('discrete', 2, 20)
    parameter.mutate = {'type': 'discrete', 'distribution': 'gauss', 'sigma': 0.5, 'min': 2, 'max': 20}
    config.run_parameters.append(parameter)

    # Parameter 3: T
    parameter = ad.Config()
    parameter.name = 'T'
    parameter.type = 'sampling'
    parameter.init = ('discrete', 1, 20)
    parameter.mutate = {'type': 'discrete', 'distribution': 'gauss', 'sigma': 0.5, 'min': 1, 'max': 20}
    config.run_parameters.append(parameter)

    # which statistics are used as a goal space
    config.goal_space_representation.type = 'statistics'
    config.goal_space_representation.config = ad.representations.static.StatisticRepresentation.default_config()
    config.goal_space_representation.config.statistics = ['activation_mass_mean',
                                                          'activation_mass_std']

    # how are goals sampled
    config.goal_selection.type = 'random'
    config.goal_selection.sampling = [(0, 1),
                                      (0, 0.4)]

    explorer = ad.explorers.GoalSpaceExplorer(system=system, config=config)
    explorer.run(5, verbose=False)

    assert len(explorer.data) == 5
    assert np.shape(explorer.statistics.reached_initial_goals) == (2, 2)  # no statistics should have been computed
    assert np.shape(explorer.statistics.target_goals) == (3, 2)  # no statistics should have been computed
    assert np.shape(explorer.statistics.reached_goals) == (3, 2)  # no statistics should have been computed


def test_specific_goal_selection():
    # use lenia as test system

    system = ad.systems.Lenia()

    config = ad.explorers.GoalSpaceExplorer.default_config()
    config.seed = 1
    config.num_of_random_initialization = 2

    # Parameter 2: R
    parameter = ad.Config()
    parameter.name = 'R'
    parameter.type = 'sampling'
    parameter.init = ('discrete', 2, 20)
    parameter.mutate = {'type': 'discrete', 'distribution': 'gauss', 'sigma': 0.5, 'min': 2, 'max': 20}
    config.run_parameters.append(parameter)

    # Parameter 3: T
    parameter = ad.Config()
    parameter.name = 'T'
    parameter.type = 'sampling'
    parameter.init = ('discrete', 1, 20)
    parameter.mutate = {'type': 'discrete', 'distribution': 'gauss', 'sigma': 0.5, 'min': 1, 'max': 20}
    config.run_parameters.append(parameter)

    # which statistics are used as a goal space
    config.goal_space_representation.type = 'statistics'
    config.goal_space_representation.config = ad.representations.static.StatisticRepresentation.default_config()
    config.goal_space_representation.config.statistics = ['activation_mass_mean',
                                                          'activation_mass_std']

    # how are goals sampled
    config.goal_selection.type = 'specific'
    config.goal_selection.goals = [[1, 2],
                                   [3, 4]]

    explorer = ad.explorers.GoalSpaceExplorer(system=system, config=config)
    explorer.run(5, verbose=False)

    assert len(explorer.data) == 5
    assert np.shape(explorer.statistics.reached_initial_goals) == (2, 2)  # no statistics should have been computed
    assert np.shape(explorer.statistics.target_goals) == (3, 2)    # no statistics should have been computed
    assert np.shape(explorer.statistics.reached_goals) == (3, 2)    # no statistics should have been computed

    # check if only defined target goals are used
    for trg_goal in explorer.statistics.target_goals:
        assert np.any([ np.all(trg_goal == x) for x in np.array([[1, 2], [3, 4]]) ])



def test_constraint_source_policy_selection():
    # use lenia as test system

    system = ad.systems.Lenia()

    config = ad.explorers.GoalSpaceExplorer.default_config()
    config.seed = 1
    config.num_of_random_initialization = 2

    # Parameter 2: R
    parameter = ad.Config()
    parameter.name = 'R'
    parameter.type = 'sampling'
    parameter.init = ('discrete', 2, 20)
    parameter.mutate = {'type': 'discrete', 'distribution': 'gauss', 'sigma': 0.5, 'min': 2, 'max': 20}
    config.run_parameters.append(parameter)

    # Parameter 3: T
    parameter = ad.Config()
    parameter.name = 'T'
    parameter.type = 'sampling'
    parameter.init = ('discrete', 1, 20)
    parameter.mutate = {'type': 'discrete', 'distribution': 'gauss', 'sigma': 0.5, 'min': 1, 'max': 20}
    config.run_parameters.append(parameter)

    # which statistics are used as a goal space
    config.goal_space_representation.type = 'statistics'
    config.goal_space_representation.config = ad.representations.static.StatisticRepresentation.default_config()
    config.goal_space_representation.config.statistics = ['activation_mass_mean',
                                                          'activation_mass_std']

    # how are goals sampled
    config.goal_selection.type = 'random'
    config.goal_selection.sampling = [(0, 1),
                                      (0, 0.4)]

    #########################################################################################
    # simple constraint

    # how source policies are selected
    config.source_policy_selection.type = 'optimal'
    config.source_policy_selection.constraints = [('id', '==', 1)]

    explorer = ad.explorers.GoalSpaceExplorer(system=system, config=config)
    explorer.run(5, verbose=False)

    assert len(explorer.data) == 5

    assert explorer.data[2].source_policy_idx == 1
    assert explorer.data[3].source_policy_idx == 1
    assert explorer.data[4].source_policy_idx == 1


    #########################################################################################
    # active settings TRUE / FALSE

    # how source policies are selected
    config.source_policy_selection.type = 'optimal'
    config.source_policy_selection.constraints = [dict(active = False,
                                                       filter = ('id', '==', 1)),
                                                  dict(active=True,
                                                       filter=('id', '==', 0))
                                                  ]

    explorer = ad.explorers.GoalSpaceExplorer(system=system, config=config)
    explorer.run(5, verbose=False)

    assert len(explorer.data) == 5

    assert explorer.data[2].source_policy_idx == 0
    assert explorer.data[3].source_policy_idx == 0
    assert explorer.data[4].source_policy_idx == 0

    #########################################################################################
    # active settings as constraint

    # how source policies are selected
    config.source_policy_selection.type = 'optimal'
    config.source_policy_selection.constraints = [dict(active = (('max', 'id'), '<', 3),
                                                       filter = ('id', '==', 1)),
                                                  dict(active = (('max', 'id'), '>=', 3),
                                                       filter=('id', '==', 0))
                                                  ]

    explorer = ad.explorers.GoalSpaceExplorer(system=system, config=config)
    explorer.run(5, verbose=False)

    assert len(explorer.data) == 5

    assert explorer.data[2].source_policy_idx == 1
    assert explorer.data[3].source_policy_idx == 1
    assert explorer.data[4].source_policy_idx == 0