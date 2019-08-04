import autodisc as ad
import numpy as np
import random
import warnings

class GoalSpaceDensityExplorer(ad.core.Explorer):
    '''
    Explorer that samples new parameters to explore based on the density in the goalspace.

    Source policies for a new exploration
    -------------------------------------

    config.source_parameter_selection
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

        # representation object
        default_config.goal_space_representation = None

        default_config.source_parameter_selection = ad.Config()
        default_config.source_parameter_selection.type = 'optimal' # either: 'optimal', 'random'
        default_config.source_parameter_selection.constraints = [] # addition constraints on the source parameters such as specific classes
        default_config.source_parameter_selection.goal_space_constraints = []  # defines constraints on the goal space for each dimension with (min, max)
        # density selection parameters
        default_config.source_parameter_selection.n_neighbors = 10

        # control over the mutation rates for the source parameters
        default_config.source_parameter_mutation = ad.Config()
        default_config.source_parameter_mutation.type = 'adaptive' # either: 'adaptive', 'fixed'
        default_config.source_parameter_mutation.target_mutation_distance = 0.1
        default_config.source_parameter_mutation.mutation_factor_learnrate = 1
        default_config.source_parameter_mutation.is_adaptive_target_mutation_distance = True
        default_config.source_parameter_mutation.mutation_distance_learnrate = 0.1

        return default_config


    def __init__(self, system, datahandler=None, config=None, **kwargs):
        super().__init__(system=system, datahandler=datahandler, config=config, **kwargs)

        # check config

        if self.config.source_parameter_selection.type not in ['random', 'optimal']:
            raise ValueError('Unknown source parameter selection type {!r} in the configuration!'.format(self.config.source_parameter_selection.type))

        if self.config.source_parameter_mutation.type not in ['adaptive', 'fixed']:
            raise ValueError('Unknown source parameter mutation type {!r} in the configuration!'.format(self.config.source_parameter_mutation.type))


        self.parameter_library = []
        self.reached_goal_library = []

        self.min_distances = []
        self.density_decision_variable = []
        self.mutation_factors = []
        self.target_mutation_distance = self.config.source_parameter_mutation.target_mutation_distance


    def get_source_parameter_idx(self):

        possible_paramters_inds = np.full(np.shape(self.reached_goal_library)[0], True)

        # apply constraints on the possible source policies if defined under config.source_parameter_selection.constraints
        if self.config.source_parameter_selection.constraints:

            for constraint in self.config.source_parameter_selection.constraints:

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
                    possible_paramters_inds = possible_paramters_inds & self.data.filter(cur_filter)


        # constraints on the goal space
        if self.config.source_parameter_selection.goal_space_constraints:

            if len(self.reached_goal_library) > 0:

                if len(self.config.source_parameter_selection.goal_space_constraints) != self.reached_goal_library.shape[1]:
                    raise ValueError('Number of dimensions of constraints for the config source_parameter_selection.goal_space_constraints ({}) must be the same as the goal space ({})',
                                     len(self.config.source_parameter_selection.goal_space_constraints),
                                     self.reached_goal_library.shape[1])

                for goal_dim_idx in range(len(self.config.source_parameter_selection.goal_space_constraints)):
                    inds = (self.reached_goal_library[:, goal_dim_idx] >= self.config.source_parameter_selection.goal_space_constraints[goal_dim_idx][0]) & \
                           (self.reached_goal_library[:, goal_dim_idx] <= self.config.source_parameter_selection.goal_space_constraints[goal_dim_idx][1])

                    possible_paramters_inds = possible_paramters_inds & inds


        if np.all(possible_paramters_inds == False):
            warnings.warn('No source parameter fullfilled the constraint. Allow all source parameters.')
            possible_paramters_inds = np.full(np.shape(self.reached_goal_library)[0], True)

        possible_paramter_idxs = np.array(list(range(np.shape(self.reached_goal_library)[0])))
        possible_paramter_idxs = possible_paramter_idxs[possible_paramters_inds]

        if self.config.source_parameter_selection.type == 'optimal':

            # identify the goal point with the lowest density
            valid_density_decision_variable = np.array(self.density_decision_variable)[possible_paramters_inds]

            max_decision_val = np.max(valid_density_decision_variable)

            max_idxs = np.where(valid_density_decision_variable == max_decision_val)[0]
            source_parameter_idx = possible_paramter_idxs[max_idxs[self.random.randint(len(max_idxs))]]

        elif self.config.source_parameter_selection.type == 'random':
            source_parameter_idx = possible_paramter_idxs[np.random.randint(len(possible_paramter_idxs))]
        else:
            raise ValueError('Unknown source policy selection type {!r} in the configuration!'.format(self.config.source_parameter_selection.type))

        return source_parameter_idx


    def update_source_parameter_selection_process(self, source_parameter_idx, reached_goal):

        if len(self.reached_goal_library) == 0:
            self.min_distances.append([])
            self.density_decision_variable = [0]
        else:

            new_distances = self.config.goal_space_representation.calc_distance(reached_goal, self.reached_goal_library)

            # identify if the new point has a smaller distance to points than their n closest neighbors
            # if yes, then add this distance to it and remove the largest
            for cur_idx in range(len(self.min_distances)):

                cur_distance = new_distances[cur_idx]

                new_min_distance_idx = None

                if len(self.min_distances[cur_idx]) < self.config.source_parameter_selection.n_neighbors:
                    new_min_distance_idx = len(self.min_distances[cur_idx])

                for neighbor_idx in range(len(self.min_distances[cur_idx]) - 1, -1, -1):

                    if self.min_distances[cur_idx][neighbor_idx] > cur_distance:
                        new_min_distance_idx = neighbor_idx
                    else:
                        break

                if new_min_distance_idx is not None:
                    self.min_distances[cur_idx].insert(new_min_distance_idx, cur_distance)

                    if len(self.min_distances[cur_idx]) > self.config.source_parameter_selection.n_neighbors:
                        self.min_distances[cur_idx].pop()

                    self.density_decision_variable[cur_idx] = np.mean(self.min_distances[cur_idx])

            # add for the new goal the min distances
            sorted_distances = np.sort(new_distances).tolist()
            sorted_distances = sorted_distances[0:self.config.source_parameter_selection.n_neighbors]

            self.min_distances.append(sorted_distances)

            self.density_decision_variable.append(np.mean(self.min_distances[-1]))

        ########################################
        # add reached goal go the library

        self.reached_goal_library = np.vstack([self.reached_goal_library, reached_goal]) if len(self.reached_goal_library) > 0 else np.array([reached_goal])

        ########################################
        # adapt the target mutation distance

        if self.config.source_parameter_mutation.is_adaptive_target_mutation_distance:

            mean_distances = np.mean(np.mean(self.min_distances, axis=1))

            if not np.isnan(mean_distances):
                self.target_mutation_distance = self.target_mutation_distance + self.config.source_parameter_mutation.mutation_distance_learnrate * (mean_distances - self.target_mutation_distance)

        ########################################
        # update the mutation factors

        if self.config.source_parameter_mutation.type == 'adaptive' and source_parameter_idx is not None:

            dist = np.linalg.norm(self.reached_goal_library[source_parameter_idx] - reached_goal)

            if dist > 0:
                self.mutation_factors[source_parameter_idx] = self.mutation_factors[source_parameter_idx] * (1 - ((1 - (self.target_mutation_distance / dist)) * self.config.source_parameter_mutation.mutation_factor_learnrate))

            self.mutation_factors.append(self.mutation_factors[source_parameter_idx])
        else:
            self.mutation_factors.append(1)


    def run(self, runs, verbose=True):

        if isinstance(runs, int):
            runs = list(range(runs))

        self.parameter_library = []
        self.reached_goal_library = []

        if verbose:
            counter = 0
            ad.gui.print_progress_bar(counter, len(runs), 'Explorations: ')

        # do n explorations
        for run_idx in runs:

            if run_idx not in self.data:

                try:
                    # set the seed if the user defined one
                    if self.config.seed is not None:
                        seed = 534 * self.config.seed + run_idx
                        self.random.seed(seed)
                        random.seed(seed) # standard random is needed for the neat sampling process
                    else:
                        seed = None
    
                    # get parameters
                    new_parameters = ad.helper.data.AttrDict() # param in parameter space (for example, contains neural network to produce initial state)
                    new_run_parameters = ad.helper.data.AttrDict()  # param that is actually given to the system (for example, the initial state that was procuded by tje source neural network)
    
                    source_parameter_idx = None
    
                    # random sampling if not enough in library
                    if len(self.parameter_library) < self.config.num_of_random_initialization:
                        # initialize the parameters
    
                        for parameter_config in self.config.run_parameters:
    
                            if parameter_config.type == 'cppn_evolution':
    
                                cppn_evo = ad.cppn.TwoDMatrixCCPNNEATEvolution(config=parameter_config['init'], matrix_size=(self.system.system_parameters.size_y, self.system.system_parameters.size_x))
                                cppn_evo.do_evolution()
    
                                if parameter_config.init.best_genome_of_last_generation:
                                    new_parameter = cppn_evo.get_best_genome_last_generation()
                                    new_run_parameter = cppn_evo.get_best_matrix_last_generation()
    
                                else:
                                    new_parameter = cppn_evo.get_best_genome()
                                    new_run_parameter = cppn_evo.get_best_matrix()
    
                            elif parameter_config.type == 'sampling':
    
                                new_parameter = ad.helper.sampling.sample_value(self.random, parameter_config['init'])
                                new_run_parameter = new_parameter
    
                            else:
                                raise ValueError('Unknown run_parameter type {!r} in configuration.'.format(parameter_config.type))
    
                            new_parameters[parameter_config['name']] = new_parameter
                            new_run_parameters[parameter_config['name']] = new_run_parameter
    
                    else:
    
                        # get source paramter which should be mutated
                        source_parameter_idx = self.get_source_parameter_idx()
                        source_parameter = self.parameter_library[source_parameter_idx]
    
                        for parameter_config in self.config.run_parameters:

                            if parameter_config.type == 'cppn_evolution':

                                # TODO: mutation factor for network mutation

                                cppn_evo = ad.cppn.TwoDMatrixCCPNNEATEvolution(init_population=source_parameter[parameter_config['name']],
                                                                               config=parameter_config['mutate'],
                                                                               matrix_size=(self.system.system_parameters.size_y, self.system.system_parameters.size_x))
                                cppn_evo.do_evolution()
    
                                if parameter_config.init.best_genome_of_last_generation:
                                    new_parameter = cppn_evo.get_best_genome_last_generation()
                                    new_run_parameter = cppn_evo.get_best_matrix_last_generation()
                                else:
                                    new_parameter = cppn_evo.get_best_genome()
                                    new_run_parameter = cppn_evo.get_best_matrix()
    
                            elif parameter_config.type == 'sampling':
    
                                new_parameter = ad.helper.sampling.mutate_value(val=source_parameter[parameter_config['name']],
                                                                                mutation_factor=self.mutation_factors[source_parameter_idx],
                                                                                rnd=self.random,
                                                                                config=parameter_config['mutate'])
                                new_run_parameter = new_parameter
    
                            else:
                                raise ValueError('Unknown run_parameter type {!r} in configuration.'.format(parameter_config.type))
    
                            new_parameters[parameter_config['name']] = new_parameter
                            new_run_parameters[parameter_config['name']] = new_run_parameter


                    #############################
                    # run experiment

                    # run with parameters
                    [observations, statistics] = self.system.run(run_parameters=new_run_parameters, stop_conditions=self.config.stop_conditions)
    
                    # get goal-space of results
                    reached_goal = self.config.goal_space_representation.calc(observations, statistics)

                    # save results
                    self.data.add_run_data(id=run_idx,
                                           seed=seed,
                                           run_parameters=new_run_parameters,
                                           observations=observations,
                                           statistics=statistics,
                                           source_parameter_idx=source_parameter_idx,   # idx of the exploration that was used as source to generate the parameters for the current exploration
                                           reached_goal=reached_goal)

                    # add policy and reached goal into the libraries
                    # do it after the run data is saved to not save them if there is an error during the saving
                    self.parameter_library.append(new_parameters)
    

                    #############################
                    # update selection processes based on the result
                    self.update_source_parameter_selection_process(source_parameter_idx, reached_goal)


                    if verbose:
                        counter += 1
                        ad.gui.print_progress_bar(counter, len(runs), 'Explorations: ')
                        if counter == len(runs):
                            print('')

                except Exception as e:
                    raise Exception('Exception during exploration run {}!'.format(run_idx)) from e
