import autodisc as ad
import numpy as np
import os
import pickle

class Classifier:

    @staticmethod
    def default_config():
        return ad.Config()


    def __init__(self, config=None, **kwargs):
        self.config = ad.config.set_default_config(kwargs, config, self.__class__.default_config())


    def calc(self, observations, statistics):
        pass


class Representation:

    @staticmethod
    def default_config():
        return ad.Config()


    def __init__(self, config=None, **kwargs):
        self.config = ad.config.set_default_config(kwargs, config, self.__class__.default_config())


    def calc(self, observations, statistics):
        pass


    def calc_distance(self, representation1, representation2):
        '''
        Standard Euclidean distance between representation vectors.
        '''

        if len(representation1) == 0 or len(representation2) == 0:
            return np.array([])

        diff = np.array(representation1) - np.array(representation2)

        if np.ndim(diff) == 1:
            dist = np.linalg.norm(diff)
        else:
            dist = np.linalg.norm(diff, axis=1)

        return dist


class System:

    @staticmethod
    def default_system_parameters():
        return ad.Config()


    @staticmethod
    def default_config():
        return ad.Config()


    def default_statistics(self):
        '''
        The default statistics associated with a system, that are loaded if the user does not specify any statistics.

        :return: List with Statistics.
        '''
        return []


    def __init__(self, statistics=None, system_parameters=None, config=None, **kwargs):
        '''
        Initialize a system.

        :param params_sys: System parameters in form of a dictionary.
        :param statistics: List of statistics.
        '''
        self.system_parameters = ad.config.set_default_config(system_parameters, self.__class__.default_system_parameters())
        self.config = ad.config.set_default_config(kwargs, config, self.__class__.default_config())
        self.run_parameters = None

        if statistics is None:
            self.statistics = self.default_statistics()
        elif not isinstance(statistics, list):
            # statistics should be a list
            self.statistics = [statistics]
        else:
            self.statistics = statistics


    def run(self, run_parameters=None, stop_conditions=100, observation_filter=None):
        '''
        Runs the system for the given parameters.

        :param run_parameters
        :param stop_conditions List with conditions by which the run stops. If any of the conditions is fullfilled, the run stops.
                               Possible conditions: int - maximum number of steps
                                                    function handle - the function is called and a bool is expected. Function parameters: system, step, state, statistics
        '''

        self.run_parameters = run_parameters

        if not isinstance(stop_conditions, list):
            stop_conditions = [stop_conditions]

        max_number_of_steps = float('inf')
        stop_condition_functions = []

        for stop_condition in stop_conditions:
            if isinstance(stop_condition, int):
                max_number_of_steps = min(stop_condition, max_number_of_steps)
            else:
                stop_condition_functions.append(stop_condition)

        states = []

        step = 0

        # intialize system
        state = self.init_run(run_parameters)
        for stat in self.statistics:
            stat.calc_after_init(self, state)
        states.append(state)

        # simulate system until a stop_condition is fullfilled
        while step < max_number_of_steps-1 and np.all([not f(self, step, state, self.statistics) for f in stop_condition_functions]):
            step += 1

            state = self.step(step)
            states.append(state)

            for stat in self.statistics:
                stat.calc_after_step(self, state, step)



        # end system
        self.stop()
        for stat in self.statistics:
            stat.calc_after_stop(self)

        # calculate the final statistics over all observations
        for stat in self.statistics:
            stat.calc_after_run(self, states)

        observations = ad.helper.data.AttrDict()
        observations.timepoints = list(range(step+1))
        observations.states = states

        # collect statistics data
        statistics_data = dict()
        for stat in self.statistics:
            statistics_data = {**stat.data, **statistics_data}
        statistics_data = ad.helper.data.AttrDict(statistics_data)

        return observations, statistics_data


    def init_run(self, run_parameters):
        pass


    def step(self, step_idx):
        pass


    def stop(self):
        pass


class SystemStatistic:

    @staticmethod
    def default_config():
        return ad.Config()


    def __init__(self, system=None, config=None, **kwargs):
        self.config = ad.config.set_default_config(kwargs, config, self.__class__.default_config())
        self.data = ad.helper.data.AttrDict()


    def reset(self):
        # set all statistics to zero
        for key in self.data.keys():
            self.data[key] = []


    def calc_after_init(self, system, obs):
        pass


    def calc_after_step(self, system, obs, step):
        pass


    def calc_after_stop(self, system):
        pass


    def calc_after_run(self, system, all_obs):
        pass




class Explorer:
    '''
    Base class for exploration experiments.

    Allows to save and load exploration results
    '''

    @staticmethod
    def default_config():
        default_config = ad.Config()
        default_config.id = None
        default_config.descr = None
        default_config.seed = None
        return default_config


    def __init__(self, system, datahandler=None, config=None, **kwargs):

        self.system = system

        # if experiment_stats is None:
        #     self.experiment_stats = []
        # if not isinstance(experiment_stats, list):
        #     self.experiment_stats = [experiment_stats]
        # else:
        #     self.experiment_stats = experiment_stats

        if datahandler is None:
            self.data = ad.ExplorationDataHandler.create(directory='./results')
        else:
            self.data = datahandler

        # set config
        self.config = ad.config.set_default_config(kwargs, config, self.__class__.default_config())

        # save some of the initial parameters in the data as documentation and to allow replication
        exploration_data = ad.DataEntry()
        exploration_data.id = self.config.id
        exploration_data.descr = self.config.descr
        exploration_data.seed = self.config.seed
        exploration_data.system_parameters = self.system.system_parameters
        exploration_data.system_name = self.system.__class__.__name__
        exploration_data.software_version = ad.__version__
        self.data.add_exploration_data(exploration_data)

        if self.config.seed is not None:
            self.random = np.random.RandomState()
        else:
            self.random = np.random.RandomState(self.config.seed)


    def run(self):
        pass


    def save(self):
        # save experiment data
        self.data.save()

        if 'directory' in self.data.config:
            self.save_explorer_obj(directory=self.data.config.directory)
        else:
            raise NotImplementedError('Saving the explorer object for non file based databases is not implemented yet!')


    def save_explorer_obj(self, directory=None):
        '''
        Saves the explorer via pickle to the given directory.

        Note, that this does not save the data and the configuration.
        Use explorer.save() to save the data and the explorer or explorer.data.save() to only save the data.
        '''

        if directory is None:
            directory = self.data.config.directory

        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath = os.path.join(directory, 'explorer.pickle')
        file = open(filepath, 'wb')

        # do not pickle the data, but save it in extra files
        tmp_data = self.data
        self.data = None

        # do not pickle configuration, because it might contain lambda functions or other element that can not be pickled
        # we assume that the configuration is anyway saved for each experiment and that the useer can load it
        tmp_config = self.config
        self.config = None

        # store configuration of the datahandler, so that it can be retrieved
        self.datahandler_config = tmp_data.config

        # pickle exploration object
        pickle.dump(self, file)

        # attach results and config again to the exploration object
        self.data = tmp_data
        del self.datahandler_config

        self.config = tmp_config


    def calc_run_statistics(self, statistics, is_rerun=False, verbose=False):
        '''Calculates the given statistics for all run data elements.'''

        if not isinstance(statistics, list):
            statistics = [statistics]
            
        if is_rerun:
            #TODO: implement rerun
            raise NotImplementedError('Rerunning the system for the calculation of extra statistics is not implemented yet!')

        system = self.system

        if verbose:
            counter = 0
            ad.gui.print_progress_bar(counter, len(self.data), 'Runs: ')

        for run_data in self.data:
            if run_data.observations is None:
                [obs,_] = system.run(run_parameters=run_data.run_parameters,
                                                         stop_conditions=self.config.stop_conditions)
                all_obs = obs.states
            else:
                all_obs = run_data.observations.states

            system.run_parameters = {**run_data.run_parameters, **self.system.system_parameters}

            for stat in statistics:
                stat.reset()
                stat.calc_after_run(system, all_obs)

            # collect statistics data
            statistics_data = dict()
            for stat in statistics:
                statistics_data = {**stat.data, **statistics_data}
            statistics_data = ad.helper.data.AttrDict(statistics_data)

            new_stats = {**statistics_data, **run_data.statistics}

            self.data.add_run_data(run_data.id, statistics=new_stats)

            if verbose:
                counter += 1
                ad.gui.print_progress_bar(counter, len(self.data), 'Runs: ')
                if counter == len(self.data):
                    print('')


    @staticmethod
    def load_explorer(directory='./results', load_data=True, run_ids=None, load_observations=None, verbose=False):

        explorer_filepath = os.path.join(directory, 'explorer.pickle')

        with open(explorer_filepath, "rb") as explorer_file:
            explorer = pickle.load(explorer_file)

        if load_data:
            explorer.data = ad.ExplorationDataHandler.create(config=explorer.datahandler_config, directory=directory)
            explorer.data.load(run_ids=run_ids, load_observations=load_observations, verbose=verbose)

        del explorer.datahandler_config

        return explorer
