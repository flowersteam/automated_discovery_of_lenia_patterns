import autodisc as ad
import numpy as np
import random
import sys
import os
import torch
from torch.utils.data import DataLoader
from autodisc.representations.static.pytorchnnrepresentation.helper import Dataset
from autodisc.representations.static.pytorchnnrepresentation import PytorchNNRepresentation
import warnings

class OnlineLearningGoalExplorer(ad.core.Explorer):
    '''Basic explorer that samples goals in a goalspace and uses a policy library to generate parameters to reach the goal.'''

    @staticmethod
    def default_config():
        default_config = ad.core.Explorer.default_config()

        default_config.stop_conditions = 200
        default_config.num_of_random_initialization = 10

        default_config.run_parameters = []

        # representation 
        default_config.goal_space_representation = ad.Config()
        default_config.goal_space_representation.type = None # either:  'pytorchnnrepresentation'

        # online_training params
        default_config.online_training = ad.Config()
        default_config.online_training.output_representation_folder = './trained_representation'
        default_config.online_training.n_runs_between_train_steps = 100
        default_config.online_training.n_epochs_per_train_steps = 40
        default_config.online_training.train_batch_size = 64
        default_config.online_training.importance_sampling_new_vs_old = 0.5
        default_config.online_training.dataset_constraints = []
        
        # goal selection 
        default_config.goal_selection = ad.Config()
        default_config.goal_selection.type = None # either: 'random', 'specific', 'function'

        default_config.source_policy_selection = ad.Config()
        default_config.source_policy_selection.type = 'optimal' # either: 'optimal', 'random'
        default_config.source_policy_selection.constraints = []
        
        
        

        return default_config


    def __init__(self, system, datahandler=None, config=None, **kwargs):
        super().__init__(system=system, datahandler=datahandler, config=config, **kwargs)

        # check config goal_space_representation
        if self.config.goal_space_representation.type not in ['pytorchnnrepresentation']:
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
        self.goal_space_representation = PytorchNNRepresentation(config = self.config.goal_space_representation.config)


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
        
        # optimizer
        learning_rate = 1e-3
        weight_decay = 1e-5
        optimizer = torch.optim.Adam(self.goal_space_representation.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # prepare output files
        if not os.path.exists(self.config.online_training.output_representation_folder):
            os.makedirs(self.config.online_training.output_representation_folder)
        
        ## subfolder to save trained models
        saved_models_folder = os.path.join(self.config.online_training.output_representation_folder, 'saved_models')
        if not os.path.exists(saved_models_folder):
            os.makedirs(saved_models_folder)
        current_weight_model_filepath = os.path.join(saved_models_folder, 'current_weight_model.pth')
        best_weight_model_filepath = os.path.join(saved_models_folder, 'best_weight_model.pth')
        
        ## subfolder with training summary
        stages_summary_folder = os.path.join(self.config.online_training.output_representation_folder, 'stages_summary')
        if not os.path.exists(stages_summary_folder):
            os.makedirs(stages_summary_folder)
            
        ## subfolder with training analysis
        training_analysis_folder = os.path.join(self.config.online_training.output_representation_folder, 'training_analysis')
        if not os.path.exists(training_analysis_folder):
            os.makedirs(training_analysis_folder)
        global_loss_train_filepath = os.path.join(training_analysis_folder, 'loss_train.csv')
        global_loss_train_file = open(global_loss_train_filepath, 'w')
        global_loss_train_file.close()
        global_loss_valid_filepath = os.path.join(training_analysis_folder, 'loss_valid.csv')
        global_loss_valid_file = open(global_loss_valid_filepath, 'w')
        global_loss_valid_file.close()
        
        ## subfolder with final goal space representation analysis
        final_representation_folder =  os.path.join(self.config.online_training.output_representation_folder, 'final_representation')
        if not os.path.exists(final_representation_folder):
            os.makedirs(final_representation_folder)
            
            
         # prepare datasets
        train_dataset = Dataset(system_state_size, data_augmentation = True)
        weights_train_dataset = [1.]
        weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_train_dataset, 1)
        train_loader = DataLoader(train_dataset, batch_size = self.config.online_training.train_batch_size, sampler = weighted_sampler)
        
        valid_dataset = Dataset(system_state_size, data_augmentation = False)
        valid_loader = DataLoader(valid_dataset, batch_size = 10)
        
        # counters
        global_epoch_idx = 0
        stage_idx = 0
        n_deads_global = 0
        n_deads_curr_stage = 0
        n_stable_animals_global = 0
        n_stable_animals_curr_stage = 0
        n_diverging_animals_global = 0
        n_diverging_animals_curr_stage = 0
        n_non_animals_global = 0
        n_non_animals_curr_stage = 0
        n_train_dataset_curr_stage = 0
        n_valid_dataset_curr_stage = 0
        last_run_idx_seen_by_nn = 0

        
        # keep track of the best result on the validation dataset
        best_valid_loss = sys.float_info.max
        
        # save a stack of last observations in memory to recompute new goals faster
        final_observations = []
        # save numerical labels for each run describing if they are {0: 'non_div_animal', 1: 'div_animal', '2': non_animal, '-1': dead} 
        labels = []
        
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
                    reached_goal = self.goal_space_representation.calc(observations)
    
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
                    
                    
                    # add data to final_observations stack
                    final_observation = torch.from_numpy(observations.states[-1]).float().unsqueeze(0) # C*H*W FloatTensor
                    final_observations.append(final_observation)
                    
                    
                    # add label describing final observation
                    if not statistics.is_dead:
                        if statistics.classifier_animal:
                            if not statistics.classifier_diverging:
                                label = 0
                                n_stable_animals_global += 1
                                n_stable_animals_curr_stage += 1
                            else:
                                label = 1
                                n_diverging_animals_global += 1
                                n_diverging_animals_curr_stage += 1
                        else:
                            label = 2
                            n_non_animals_global +=1
                            n_non_animals_curr_stage +=1
                                
                    else:
                        label = -1
                        n_deads_global += 1
                        n_deads_curr_stage += 1
                    labels.append(label)
                        
                    # if run_idx % n_runs_between_train_steps == 0: run a new stage of training
                    if ((run_idx+1) % self.config.online_training.n_runs_between_train_steps == 0):
                        ''' Filter samples '''
                        constrained_run_inds = np.full(np.shape(self.goal_library)[0], True)
                        # apply constraints on the last runs to train network on a specific type of image
                        if self.config.online_training.dataset_constraints:
                            for constraint in self.config.online_training.dataset_constraints:
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
                                    constrained_run_inds = constrained_run_inds & self.data.filter(cur_filter)
                
                        if np.sum(constrained_run_inds) < 10:
                            warnings.warn('Not enough runs fullfilled the constraint to start training, skipping the init stage')
                            
                        else:          
                            ''' If enough samples start stage of training '''
                            
                            '''prepare output files'''
                            stage_epoch_idx = 0
                            ## saved_models
                            curr_stage_weight_model_filepath = os.path.join (saved_models_folder, 'stage_{:06d}_weight_model.pth'.format(stage_idx))
                            
                            ## stage_summary
                            curr_stage_summary_folder = os.path.join (stages_summary_folder, 'stage_{:06d}'.format(stage_idx))
                            if not os.path.exists(curr_stage_summary_folder):
                                os.makedirs(curr_stage_summary_folder)
                            curr_stage_summary_filepath = os.path.join (curr_stage_summary_folder, 'summary.csv')
                            curr_stage_train_dataset_filepath = os.path.join (curr_stage_summary_folder, 'train_dataset.npz')
                            curr_stage_valid_dataset_filepath = os.path.join (curr_stage_summary_folder, 'valid_dataset.npz')
                            
                            ''' Update training and validation datasets and weights '''
                            last_constrained_run_idxs = np.array(list(range(last_run_idx_seen_by_nn + 1, np.shape(self.goal_library)[0])))
                            last_constrained_run_idxs = last_constrained_run_idxs[constrained_run_inds[last_run_idx_seen_by_nn + 1:]]
                            
                            # iterate to new discoverues and add it to train/valid dataset
                            run_ids_added_to_train_dataset = []
                            run_ids_added_to_valid_dataset = []
                            for run_id in last_constrained_run_idxs:
                                if (train_loader.dataset.n_images + valid_loader.dataset.n_images + 1) % 10 == 0:
                                    valid_loader.dataset.images.append(final_observations[run_id])
                                    valid_loader.dataset.labels.append(labels[run_id])
                                    valid_loader.dataset.n_images += 1
                                    n_valid_dataset_curr_stage += 1
                                    run_ids_added_to_valid_dataset.append(run_id)
                                else:
                                    train_loader.dataset.images.append(final_observations[run_id])
                                    train_loader.dataset.labels.append(labels[run_id])
                                    train_loader.dataset.n_images += 1
                                    n_train_dataset_curr_stage += 1
                                    run_ids_added_to_train_dataset.append(run_id)
                            
                            
                            # update weight
                            if stage_idx == 0:
                                weights = [1.0 / train_loader.dataset.n_images] * (train_loader.dataset.n_images)
                                weighted_sampler.num_samples = len(weights)
                                weighted_sampler.weights = torch.tensor(weights, dtype=torch.double)
                            else:
                                weights = [(1.0 - self.config.online_training.importance_sampling_new_vs_old) / (train_loader.dataset.n_images - n_train_dataset_curr_stage)] * (train_loader.dataset.n_images - n_train_dataset_curr_stage)
                                if n_train_dataset_curr_stage > 0:
                                    weights += ([self.config.online_training.importance_sampling_new_vs_old / (n_train_dataset_curr_stage) ] * n_train_dataset_curr_stage)
                                weighted_sampler.num_samples = len(weights)
                                weighted_sampler.weights = torch.tensor(weights, dtype=torch.double)
                            
                            ''' training stage: loop over epochs '''  
                            for epoch in range(self.config.online_training.n_epochs_per_train_steps):
                                # training epoch
                                train_losses = self.goal_space_representation.train_epoch(train_loader, optimizer)
                                
                                # validation epoch
                                valid_losses =  self.goal_space_representation.valid_epoch (valid_loader)
    
                                # output files
                                ## training_analysis
                                global_loss_train_file = open(global_loss_train_filepath, 'a')
                                global_loss_train_file.write("Epoch: {0}".format(global_epoch_idx))
                                for k, v in train_losses.items():
                                    global_loss_train_file.write("\t{0}: {1:.6f}".format(k,v))
                                global_loss_train_file.write("\n")
                                global_loss_train_file.close()
                                global_loss_valid_file = open(global_loss_valid_filepath, 'a')
                                global_loss_valid_file.write("{0}".format(global_epoch_idx))
                                for k, v in valid_losses.items():
                                    global_loss_valid_file.write("\t{0}: {1:.6f}".format(k,v))
                                global_loss_valid_file.write("\n")  
                                global_loss_valid_file.close()
                                
                                # save current weight 
                                self.goal_space_representation.save_model(global_epoch_idx, optimizer, current_weight_model_filepath)
                                # save best weight
                                valid_loss = valid_losses['total']
                                if valid_loss < best_valid_loss:
                                    best_valid_loss = valid_loss
                                    self.goal_space_representation.save_model(global_epoch_idx, optimizer, best_weight_model_filepath)
                                
                                # update epoch counter
                                stage_epoch_idx += 1
                                global_epoch_idx += 1
                            
                            ''' save stage network'''
                            self.goal_space_representation.save_model(global_epoch_idx, optimizer, curr_stage_weight_model_filepath)
                            
                            ''' save stage datasets '''
                            train_loader.dataset.save(curr_stage_train_dataset_filepath)
                            valid_loader.dataset.save(curr_stage_valid_dataset_filepath)
                            
                            ''' stage summary '''
                            curr_stage_training_file = open(curr_stage_summary_filepath, 'w')
                            curr_stage_training_file.write('variable: \t new \t / \t tot \n')
                            curr_stage_training_file.write('n_epochs: \t {0} \t / \t {1} \n'.format(stage_epoch_idx, global_epoch_idx))
                            curr_stage_training_file.write('n_explored: \t {0} \t / \t {1} \n'.format(self.config.online_training.n_runs_between_train_steps, len(self.policy_library)))
                            curr_stage_training_file.write('n_deads: \t {0} \t / \t {1} \n'.format(n_deads_curr_stage, n_deads_global))
                            curr_stage_training_file.write('n_stable_animals: \t {0} \t / \t {1} \n'.format(n_stable_animals_curr_stage, n_stable_animals_global))
                            curr_stage_training_file.write('n_diverging_animals: \t {0} \t / \t {1} \n'.format(n_diverging_animals_curr_stage, n_diverging_animals_global))
                            curr_stage_training_file.write('n_non_animals: \t {0} \t / \t {1} \n'.format(n_non_animals_curr_stage, n_non_animals_global))
                            curr_stage_training_file.write('n_train_dataset: \t {0} \t / \t {1} \n'.format(n_train_dataset_curr_stage, train_loader.dataset.n_images))
                            curr_stage_training_file.write('n_valid_dataset: \t {0} \t / \t {1} \n'.format(n_valid_dataset_curr_stage, valid_loader.dataset.n_images))
                            curr_stage_training_file.write('importance_weight_new_dataset: \t {0} \t / \t {1} \n'.format(self.config.online_training.importance_sampling_new_vs_old, 1.0 - self.config.online_training.importance_sampling_new_vs_old))
                            curr_stage_training_file.write('\n\n')
                            curr_stage_training_file.write('new_run_ids_added_to_train_dataset: \t {0} \n'.format(run_ids_added_to_train_dataset))
                            curr_stage_training_file.write('new_run_ids_added_to_valid_dataset: \t {0} \n'.format(run_ids_added_to_valid_dataset))
                            curr_stage_training_file.close()
    
                            ''' Update stage counter'''
                            n_deads_curr_stage = 0
                            n_stable_animals_curr_stage = 0
                            n_diverging_animals_curr_stage = 0
                            n_non_animals_curr_stage = 0
                            n_train_dataset_curr_stage = 0
                            n_valid_dataset_curr_stage = 0
                            stage_idx += 1
                            last_run_idx_seen_by_nn = run_idx
                            
                            
                            ''' Update reached goal of old runs after new stage'''
                             # save updated results 
                                ##/!\ we dont save it in the outputs run_data_* files where the reached_goal corresponds to the one in the current goal space at the time of exploration
                                ## we compute new goals by batches as for training to go faster
                            n_old_runs = len(self.goal_library)
                            n_full_batches = n_old_runs // self.config.online_training.train_batch_size
                            self.goal_space_representation.model.eval()
                            with torch.no_grad():
                                for batch_idx in range(n_full_batches):
                                    input_observations = torch.stack(final_observations[(batch_idx * self.config.online_training.train_batch_size) : ((batch_idx + 1) * self.config.online_training.train_batch_size)], dim=0)
                                    output_goals = self.goal_space_representation.model.calc(input_observations)
                                    for idx in range(self.config.online_training.train_batch_size):
                                        self.goal_library[(batch_idx * self.config.online_training.train_batch_size) + idx] = output_goals[idx].squeeze(0).cpu().numpy()
                                # last batch with remaining indexes:
                                n_remaining_idx = n_old_runs - n_full_batches * self.config.online_training.train_batch_size
                                if n_remaining_idx > 0:
                                    input_observations = torch.stack(final_observations[(n_full_batches * self.config.online_training.train_batch_size) :], dim=0)
                                    output_goals = self.goal_space_representation.model.calc(input_observations)
                                    for idx in range(n_remaining_idx):
                                        self.goal_library[(n_full_batches * self.config.online_training.train_batch_size) + idx] = output_goals[idx].squeeze(0).cpu().numpy()
                                
                                
                                
                            ''' Save explorer '''
                            self.save()
                        
                    
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
        
        ''' plot training evolution '''
        plot_training_loss_evolution(training_analysis_folder)
        plot_training_dataset_evolution(stages_summary_folder)
        
        ''' final representation accuracy on valid dataset '''
        # accuracy on valid dataset
        validation_accuracy_folder = os.path.join(final_representation_folder, 'validation_accuracy')
        if not os.path.exists(validation_accuracy_folder):
            os.makedirs(validation_accuracy_folder)
        validation_reconstruction_images_folder = os.path.join(validation_accuracy_folder, 'reconstruction_images')
        if not os.path.exists(validation_reconstruction_images_folder):
            os.makedirs(validation_reconstruction_images_folder)
        valid_losses = self.goal_space_representation.valid_epoch_and_save_images(valid_loader, validation_reconstruction_images_folder)
        analyse_final_representation_accuracy(valid_losses, validation_accuracy_folder)
          


import matplotlib.pyplot as plt
plt.ioff()
from glob import glob
import re
  
def plot_training_loss_evolution(output_folder):
    # plot train and valid loss curves
    train_losses = {}
    with open(os.path.join(output_folder, 'loss_train.csv'), 'r') as f:
        lineslist = [line.rstrip() for line in f]
        n_epochs = len(lineslist)
        for line in lineslist:
            line = line.split('\t')
            for col in range(1, len(line)):
                k,v = line[col].split(' ')
                k = k[:-1]
                if k not in train_losses:
                        train_losses[k] = [float(v)]
                else:
                    train_losses[k].append(float(v))
            
    valid_losses = {}
    with open(os.path.join(output_folder, 'loss_valid.csv'), 'r') as f:
        lineslist = [line.rstrip() for line in f]
        for line in lineslist:
            line = line.split('\t')
            for col in range(1, len(line)):
                k,v = line[col].split(' ')
                k = k[:-1]
                if k not in valid_losses:
                        valid_losses[k] = [float(v)]
                else:
                    valid_losses[k].append(float(v))
                
    for key in train_losses:
        output_filename = os.path.join(output_folder, '{}_curves.png'.format(key))
        fig, ax = plt.subplots()
        ax.plot(range(n_epochs), train_losses[key], 'k', color='red', label='train data')
        ax.plot(range(n_epochs), valid_losses[key], 'k', color='green', label='valid data')
        ax.set_xlabel('train epoch')
        ax.set_ylabel('{}'.format(key))
        ax.legend(loc='upper center', shadow=True)
        plt.savefig(output_filename)
        plt.close()
    
    return

def plot_training_dataset_evolution(output_folder):
    # plot dataset evolution
    output_filename = os.path.join(output_folder, 'dataset_summary.png')
    
    file_matches = glob(os.path.join(output_folder, 'stage_*/summary.csv'))
    last_stage_id = 0
    for match in file_matches:
        id_as_str = re.findall('_(\d+)', match)
        if len(id_as_str) > 0:
            id_as_int = int(id_as_str[0])
            if id_as_int > last_stage_id:
                last_stage_id = id_as_int
        
    n_stages = last_stage_id + 1
    n_explored = []
    n_deads = []
    n_stable_animals = []
    n_diverging_animals = []
    n_non_animals = []
    n_valid_dataset = []
    n_train_dataset = []

    for stage_idx in range(n_stages):
        training_summary_filename = os.path.join(output_folder, 'stage_{:06d}/summary.csv'.format(stage_idx))
        with open(training_summary_filename, 'r') as f:
            lineslist = [line.rstrip() for line in f]
            for line in lineslist:
                line = line.split('\t')
                if line[0] == 'n_explored: ':
                    n_explored.append(float(line[3]))
                if line[0] == 'n_deads: ':
                    n_deads.append(float(line[3]))
                if line[0] == 'n_stable_animals: ':
                    n_stable_animals.append(float(line[3]))
                if line[0] == 'n_diverging_animals: ':
                    n_diverging_animals.append(float(line[3]))
                if line[0] == 'n_non_animals: ':
                    n_non_animals.append(float(line[3]))
                if line[0] == 'n_train_dataset: ':
                    n_train_dataset.append(float(line[3]))
                if line[0] == 'n_valid_dataset: ':
                    n_valid_dataset.append(float(line[3]))
            
    # plot train and test loss
    fig, ax = plt.subplots()
    ax.plot(range(n_stages), n_explored, label='n_explored')
    ax.text(n_stages-1, n_explored[-1], str(n_explored[-1]))
    ax.plot(range(n_stages), n_deads, label='n_deads')
    ax.text(n_stages-1, n_deads[-1], str(n_deads[-1]))
    ax.plot(range(n_stages), n_stable_animals, label='n_stable_animals')
    ax.text(n_stages-1, n_stable_animals[-1], str(n_stable_animals[-1]))
    ax.plot(range(n_stages), n_diverging_animals, label='n_diverging_animals')
    ax.text(n_stages-1, n_diverging_animals[-1], str(n_diverging_animals[-1]))
    ax.plot(range(n_stages), n_non_animals, label='n_non_animals')
    ax.text(n_stages-1, n_non_animals[-1], str(n_non_animals[-1]))
    ax.plot(range(n_stages), n_train_dataset, label='n_train_dataset')
    ax.text(n_stages-1, n_train_dataset[-1], str(n_train_dataset[-1]))
    ax.plot(range(n_stages), n_valid_dataset, label='n_valid_dataset')
    ax.text(n_stages-1, n_valid_dataset[-1], str(n_valid_dataset[-1]))
    
    ax.set_ylabel('# dataset')
    ax.set_xlabel('stage idx')
    ax.legend(loc='upper center', shadow=True)
    plt.savefig(output_filename)
    plt.close()
    return


def analyse_final_representation_accuracy(valid_losses, validation_accuracy_folder):    
    # sorted classification file
    if 'CE' in valid_losses:
        indexes_sorted_by_CE_loss = np.argsort(valid_losses['CE']).astype('int')
        sorted_classification_filename = os.path.join(validation_accuracy_folder, 'sorted_classification_cases.csv')
        with open(sorted_classification_filename, 'w') as f:
            for idx in indexes_sorted_by_CE_loss:
                f.write('{} \t {:.4f} \n'.format(idx, valid_losses['CE'][idx]))
    
    # sorted reconstruction file
    if 'BCE' in valid_losses:
        indexes_sorted_by_BCE_loss = np.argsort(valid_losses['BCE']).astype('int')
        sorted_reconstruction_filename = os.path.join(validation_accuracy_folder, 'sorted_reconstruction_cases.csv')
        with open(sorted_reconstruction_filename, 'w') as f:
            for idx in indexes_sorted_by_BCE_loss:
                f.write('{} \t {:.4f} \n'.format(idx, valid_losses['BCE'][idx]))

    
    # statistics file
    statistics_filename =  os.path.join(validation_accuracy_folder, 'statistics.csv')
    with open(statistics_filename, 'w') as f:
        f.write('\n\n')
        f.write('  \t Mean \t Med \t Std \t Min \t Max \t 90-percentile \n')
        for k in valid_losses:
            f.write(' {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \n'.format(k, np.mean(valid_losses[k]), np.median(valid_losses[k]), np.std(valid_losses[k]), np.min(valid_losses[k]), np.max(valid_losses[k]), np.percentile(valid_losses[k], 90) ))
        f.write('\n\n')
    
    return