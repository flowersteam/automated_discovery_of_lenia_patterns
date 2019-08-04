import autodisc as ad


def get_system_parameters():
    system_parameters = ad.systems.Lenia.default_system_parameters()
    system_parameters.size_y = 256
    system_parameters.size_x = 256

    return system_parameters


def get_explorer_config():

    explorer_config = ad.explorers.OnlineLearningGoalExplorer.default_config()
    explorer_config.seed = <repetition_id>
    explorer_config.num_of_random_initialization = <n_init>

    explorer_config.run_parameters = []

    # Parameter 1: init state
    parameter = ad.Config()
    parameter.name = 'init_state'
    parameter.type = 'cppn_evolution'

    parameter.init = ad.cppn.TwoDMatrixCCPNNEATEvolution.default_config()
    parameter.init.neat_config_file = 'neat_config.cfg'
    parameter.init.n_generations = 1
    parameter.init.best_genome_of_last_generation = True

    parameter.mutate = ad.cppn.TwoDMatrixCCPNNEATEvolution.default_config()
    parameter.mutate.neat_config_file = 'neat_config.cfg'
    parameter.mutate.n_generations = 2
    parameter.mutate.best_genome_of_last_generation = True

    explorer_config.run_parameters.append(parameter)

    # Parameter 2: R
    parameter = ad.Config()
    parameter.name = 'R'
    parameter.type = 'sampling'
    parameter.init = ('discrete', 2, 20)
    parameter.mutate = {'type': 'discrete', 'distribution': 'gauss', 'sigma': <r_sigma>, 'min': 2, 'max': 20}
    explorer_config.run_parameters.append(parameter)

    # Parameter 3: T
    parameter = ad.Config()
    parameter.name = 'T'
    parameter.type = 'sampling'
    parameter.init = ('discrete', 1, 20)
    parameter.mutate = {'type': 'discrete', 'distribution': 'gauss', 'sigma': <t_sigma>, 'min': 1, 'max': 20}
    explorer_config.run_parameters.append(parameter)

    # Parameter 4: b
    parameter = ad.Config()
    parameter.name = 'b'
    parameter.type = 'sampling'
    parameter.init = ('function', ad.helper.sampling.sample_vector, (('discrete', 1, 3), (0, 1)))
    parameter.mutate = {'type': 'continuous', 'distribution': 'gauss', 'sigma': <b_sigma>, 'min': 0, 'max': 1}
    explorer_config.run_parameters.append(parameter)

    # Parameter 5: m
    parameter = ad.Config()
    parameter.name = 'm'
    parameter.type = 'sampling'
    parameter.init = ('continuous', 0, 1)
    parameter.mutate = {'type': 'continuous', 'distribution': 'gauss', 'sigma': <m_sigma>, 'min': 0, 'max': 1}
    explorer_config.run_parameters.append(parameter)

    # Parameter 6: s
    parameter = ad.Config()
    parameter.name = 's'
    parameter.type = 'sampling'
    parameter.init = ('continuous', 0.001, 0.3)
    parameter.mutate = {'type': 'continuous', 'distribution': 'gauss', 'sigma': <s_sigma>, 'min': 0.001, 'max': 0.3}
    explorer_config.run_parameters.append(parameter)

    # which statistics are used as a goal space
    explorer_config.goal_space_representation.type = 'pytorchnnrepresentation'
    explorer_config.goal_space_representation.config = ad.representations.static.PytorchNNRepresentation.default_config()
    explorer_config.goal_space_representation.config.initialization = ad.Config()
    explorer_config.goal_space_representation.config.initialization.type = '<goal_space_nn_initialization_type>'
    explorer_config.goal_space_representation.config.initialization.load_from_model_path = '<goal_space_nn_load_from_model_path>'
    explorer_config.goal_space_representation.config.initialization.model_type = '<goal_space_nn_model_type>'
    system_params = get_system_parameters()
    explorer_config.goal_space_representation.config.initialization.model_init_params = {<goal_space_nn_init_params>, 'input_size': (system_params.size_y, system_params.size_x), 'num_classes': 3}

    #online training parameters
    explorer_config.online_training.output_representation_folder = './trained_representation'
    explorer_config.online_training.n_runs_between_train_steps = 100
    explorer_config.online_training.n_epochs_per_train_steps = 40
    explorer_config.online_training.train_batch_size = 64
    explorer_config.online_training.importance_sampling_new_vs_old = <online_training_importance_sampling_new>
    explorer_config.online_training.dataset_constraints  = [<online_training_dataset_constraints>]

    # how are goals sampled
    explorer_config.goal_selection.type = 'random'
    explorer_config.goal_selection.sampling = [<goal_sampling>] * explorer_config.goal_space_representation.config.initialization.model_init_params['n_latents']


    # how are the source policies for a mutation are selected
    explorer_config.source_policy_selection.type = '<src_policy_selection>'

    return explorer_config

def get_number_of_explorations():
    return <n_exp>
