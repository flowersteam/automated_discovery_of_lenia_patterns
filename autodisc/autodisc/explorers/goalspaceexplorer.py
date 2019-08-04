import autodisc as ad
import numpy as np
import random
import warnings


class GoalSpaceExplorer(ad.core.Explorer):
    '''
    Basic explorer that samples goals in a goalspace and uses a policy library to generate parameters to reach the goal.

    Source policies for a new exploration
    -------------------------------------

    config.source_policy_selection
        .type: 'optimal' or 'random'.
                Optimal selects a previous exploration which has the closest point in the goal space to the new goal.
                Random selects a random previous exploration as source.

        .constraints: Can be used to define a constraints on the source policies based on filters.
                      Is a list of filters or dictionaries with the following properties:
                            active: Defines if and when a constraint is active.
                                    Can be 'True', 'False', or a condition.
                            filter: Definition of the filter that defines which previous exploration runs are allowed as source.

    Examples of constraints:

        Only explorations for which statistics.classifier_animal is True:
            dict(active = True,
                 filter = ('statistics.classifier_animal', '==', True))

        Only explorations for which statistics.is_dead is False and statistics.classifier_animal is False:
            dict(active = True,
                 filter = (('statistics.is_dead', '==', False), 'and', ('statistics.classifier_animal', '==', False))

        Only active after 100 animals have been discovered:
            dict(active = (('sum', 'statistics.classifier_animal'), '>=', 100),
                 filter = (('statistics.is_dead', '==', False), 'and', ('statistics.classifier_animal', '==', False))
    '''


    @staticmethod
    def default_config():
        default_config = ad.core.Explorer.default_config()

        default_config.stop_conditions = 200
        default_config.num_of_random_initialization = 10

        default_config.run_parameters = []

        # TODO: allow the definition of arbitrary representation spaces
        default_config.goal_space_representation = ad.Config()
        default_config.goal_space_representation.type = None # either:  'pytorchnnrepresentation' or 'statisticsrepresentation' or 'functionrepresentation'

        default_config.goal_selection = ad.Config()
        default_config.goal_selection.type = None # either: 'random', 'specific', 'function'

        default_config.source_policy_selection = ad.Config()
        default_config.source_policy_selection.type = 'optimal' # either: 'optimal', 'random'
        default_config.source_policy_selection.constraints = []

        return default_config


    def __init__(self, system, datahandler=None, config=None, **kwargs):
        super().__init__(system=system, datahandler=datahandler, config=config, **kwargs)

        # check config goal_space_representation
        if self.config.goal_space_representation.type not in ['statisticsrepresentation', 'statistics', 'pytorchnnrepresentation', 'pytorchnn', 'functionrepresentation', 'function']:
            raise ValueError('Unknown goal space representation type {!r} in the configuration!'.format(self.config.goal_space_representation.type))
        
        # initialize goal_space_representation
        self.goal_space_representation = None

        # check config goal_selection     
        if self.config.goal_selection.type not in ['random', 'specific', 'function']:
            raise ValueError('Unknown goal generation type {!r} in the configuration!'.format(self.config.goal_selection.type))

        self.policy_library = []
        self.goal_library = []

        self.statistics = ad.helper.data.AttrDict()
        self.statistics.target_goals = []
        self.statistics.reached_goals = []
        self.statistics.reached_initial_goals = []


    def load_goal_space_representation(self):
        if self.config.goal_space_representation.type in ['statisticsrepresentation', 'statistics']:
            self.goal_space_representation = ad.representations.static.StatisticRepresentation(config = self.config.goal_space_representation.config)
        elif self.config.goal_space_representation.type in ['pytorchnnrepresentation', 'pytorchnn']:
            self.goal_space_representation = ad.representations.static.PytorchNNRepresentation(config = self.config.goal_space_representation.config)
        elif self.config.goal_space_representation.type in ['functionrepresentation', 'function']:
            self.goal_space_representation = ad.representations.FunctionRepresentation(config = self.config.goal_space_representation.config)

        
    def get_next_goal(self):
        '''Defines the next goal of the exploration.'''

        if self.config.goal_selection.type == 'random':

            target_goal = np.zeros(len(self.config.goal_selection.sampling))
            for idx, sampling_config in enumerate(self.config.goal_selection.sampling):
                target_goal[idx] = ad.helper.sampling.sample_value(self.random, sampling_config)

        elif self.config.goal_selection.type == 'specific':

            if np.ndim(self.config.goal_selection.goals) == 1:
                target_goal = np.array(self.config.goal_selection.goals)
            else:
                rand_idx = self.random.randint(np.shape(self.config.goal_selection.goals)[0])
                target_goal = np.array(self.config.goal_selection.goals[rand_idx])

        elif self.config.goal_selection.type == 'function':

            if 'config' in self.config.goal_selection:
                target_goal = self.config.goal_selection.function(self, self.config.goal_space, self.config.goal_selection.config)
            else:
                target_goal = self.config.goal_selection.function(self, self.config.goal_space)

        else:
            raise ValueError('Unknown goal generation type {!r} in the configuration!'.format(self.config.goal_selection.type))

        return target_goal


    def get_source_policy_idx(self, target_goal):

        possible_run_inds = np.full(np.shape(self.goal_library)[0], True)

        # apply constraints on the possible source policies if defined under config.source_policy_selection.constraints
        if self.config.source_policy_selection.constraints:

            for constraint in self.config.source_policy_selection.constraints:

                if isinstance(constraint, tuple):
                    # if tuple, then this is the contraint and it is active
                    cur_is_active = True
                    cur_filter = constraint
                else:
                    # otherwise assume it is a dict/config with the fields: active, filter
                    if 'active' not in constraint:
                        cur_is_active = True
                    else:
                        if isinstance(constraint['active'], tuple):
                            cur_is_active = self.data.filter(constraint['active'])
                        else:
                            cur_is_active = constraint['active']

                    cur_filter = constraint['filter']

                if cur_is_active:
                    possible_run_inds = possible_run_inds & self.data.filter(cur_filter)

        if np.all(possible_run_inds == False):
            warnings.warn('No policy fullfilled the constraint. Allow all policies.')
            possible_run_inds = np.full(np.shape(self.goal_library)[0], True)

        possible_run_idxs = np.array(list(range(np.shape(self.goal_library)[0])))
        possible_run_idxs = possible_run_idxs[possible_run_inds]

        if self.config.source_policy_selection.type == 'optimal':

            # get distance to other goals
            goal_distances = self.goal_space_representation.calc_distance(target_goal, self.goal_library[possible_run_inds,:])

            # select goal with minimal distance
            source_policy_idx = possible_run_idxs[np.argmin(goal_distances)]

        elif self.config.source_policy_selection.type == 'random':
            source_policy_idx = possible_run_idxs[np.random.randint(len(possible_run_idxs))]
        else:
            raise ValueError('Unknown source policy selection type {!r} in the configuration!'.format(self.config.source_policy_selection.type))

        return source_policy_idx


    def run(self, runs, verbose=True):

        if isinstance(runs, int):
            runs = list(range(runs))

        self.policy_library = []
        self.goal_library = []

        self.statistics.target_goals = []
        self.statistics.reached_goals = []
        self.statistics.reached_initial_goals = []

        system_state_size = (self.system.system_parameters.size_y, self.system.system_parameters.size_x)

        # if not loaded create goal space representation
        if self.goal_space_representation is None:
            self.load_goal_space_representation()
            
        if verbose:
            counter = 0
            ad.gui.print_progress_bar(counter, len(runs), 'Explorations: ')

        # create the system:
        for run_idx in runs:

            if run_idx not in self.data:

                try:
                    # set the seed if the user defined one
                    if self.config.seed is not None:
                        seed = 100000 * self.config.seed + run_idx
                        self.random.seed(seed)
                        random.seed(seed) # standard random is needed for the neat sampling process
                    else:
                        seed = None
    
                    target_goal = []
    
                    # get a policy - run_parameters
                    policy_parameters = ad.helper.data.AttrDict()
                    run_parameters = ad.helper.data.AttrDict()
    
                    source_policy_idx = None
    
                    # random sampling if not enough in library
                    if len(self.policy_library) < self.config.num_of_random_initialization:
                        # initialize the parameters
    
                        for parameter_config in self.config.run_parameters:
    
                            if parameter_config.type == 'cppn_evolution':
    
                                cppn_evo = ad.cppn.TwoDMatrixCCPNNEATEvolution(config=parameter_config['init'], matrix_size=system_state_size)
                                cppn_evo.do_evolution()
    
                                if parameter_config.init.best_genome_of_last_generation:
                                    policy_parameter = cppn_evo.get_best_genome_last_generation()
                                    run_parameter = cppn_evo.get_best_matrix_last_generation()
    
                                else:
                                    policy_parameter = cppn_evo.get_best_genome()
                                    run_parameter = cppn_evo.get_best_matrix()
    
                            elif parameter_config.type == 'sampling':
    
                                policy_parameter = ad.helper.sampling.sample_value(self.random, parameter_config['init'])
                                run_parameter = policy_parameter
    
                            else:
                                raise ValueError('Unknown run_parameter type {!r} in configuration.'.format(parameter_config.type))
    
                            policy_parameters[parameter_config['name']] = policy_parameter
                            run_parameters[parameter_config['name']] = run_parameter
    
                    else:
    
                        # sample a goal space from the goal space
                        target_goal = self.get_next_goal()
    
                        # get source policy which should be mutated
                        source_policy_idx = self.get_source_policy_idx(target_goal)
                        source_policy = self.policy_library[source_policy_idx]
    
                        for parameter_config in self.config.run_parameters:
    
                            if parameter_config.type == 'cppn_evolution':
    
                                cppn_evo = ad.cppn.TwoDMatrixCCPNNEATEvolution(init_population=source_policy[parameter_config['name']],
                                                                               config=parameter_config['mutate'], matrix_size=system_state_size)
                                cppn_evo.do_evolution()
    
                                if parameter_config.init.best_genome_of_last_generation:
                                    policy_parameter = cppn_evo.get_best_genome_last_generation()
                                    run_parameter = cppn_evo.get_best_matrix_last_generation()
                                else:
                                    policy_parameter = cppn_evo.get_best_genome()
                                    run_parameter = cppn_evo.get_best_matrix()
    
                            elif parameter_config.type == 'sampling':
    
                                policy_parameter = ad.helper.sampling.mutate_value(val=source_policy[parameter_config['name']],
                                                                                   rnd=self.random,
                                                                                   config=parameter_config['mutate'])
                                run_parameter = policy_parameter
    
                            else:
                                raise ValueError('Unknown run_parameter type {!r} in configuration.'.format(parameter_config.type))
    
                            policy_parameters[parameter_config['name']] = policy_parameter
                            run_parameters[parameter_config['name']] = run_parameter
    
                    # run with parameters
                    [observations, statistics] = self.system.run(run_parameters=run_parameters, stop_conditions=self.config.stop_conditions)
    
                    # get goal-space of results
                    reached_goal = self.goal_space_representation.calc(observations, statistics)
    
                    # save results
                    self.data.add_run_data(id=run_idx,
                                           seed=seed,
                                           run_parameters=run_parameters,
                                           observations=observations,
                                           statistics=statistics,
                                           source_policy_idx=source_policy_idx,   # idx of the exploration that was used as source to generate the parameters for the current exploration
                                           target_goal=target_goal,
                                           reached_goal=reached_goal)
    
                    # add policy and reached goal into the libraries
                    # do it after the run data is saved to not save them if there is an error during the saving
                    self.policy_library.append(policy_parameters)
    
                    if len(self.goal_library) <= 0:
                        self.goal_library = np.array([reached_goal])
                    else:
                        self.goal_library = np.vstack([self.goal_library, reached_goal])
    
                    # save statistics
                    if len(target_goal) <= 0:
                        self.statistics.reached_initial_goals.append(reached_goal)
                    else:
                        self.statistics.target_goals.append(target_goal)
                        self.statistics.reached_goals.append(reached_goal)
    
                    if verbose:
                        counter += 1
                        ad.gui.print_progress_bar(counter, len(runs), 'Explorations: ')
                        if counter == len(runs):
                            print('')

                except Exception as e:

                    # save statistics in case of error
                    self.data.add_exploration_data(statistics=self.statistics)

                    raise Exception('Exception during exploration run {}!'.format(run_idx)) from e

        # save statistics
        self.data.add_exploration_data(statistics=self.statistics)
