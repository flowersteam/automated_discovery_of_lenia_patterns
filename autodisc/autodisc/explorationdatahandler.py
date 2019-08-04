import autodisc as ad
from collections import OrderedDict
import os
import json
import numpy as np
from glob import glob
import re
import warnings


class DataEntry(ad.helper.data.AttrDict):
    pass


class RunDataEntry(ad.helper.data.AttrDict):
    '''Allows to load observations if they are accessed by the user.'''

    def __init__(self, datahandler, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.datahandler = datahandler


    def __getitem__(self, key):
        val = super().__getitem__(key)

        # if observation does not exists and should be loaded --> load it
        if key == 'observations' and val is None and self.datahandler.config.load_observations:
            obs = self.datahandler.load_observations(self.id, verbose=False)
            val = obs[self.id]

        return val


    def __getattr__(self, key):
        val = super().__getattr__(key)

        # if observation does not exists and should be loaded --> load it
        if key == 'observations' and val is None and self.datahandler.config.load_observations:
            obs = self.datahandler.load_observations(self.id, verbose=False)
            val = obs[self.id]

        return val


    def is_observations_loaded(self):
        'Allows to check if observations are loaded without triggering the loading of the observations.'

        if not super().__contains__('observations'):
            return False
        else:
            val = super().__getitem__('observations')

            if val is None:
                return False
            else:
                return True




class ExplorationDataHandler:
    '''
    Base of all Database classes.
    '''

    @staticmethod
    def default_config():

        default_config = ad.Config()
        default_config.db_type = 'file_repetition' # 'file_repetition', 'memory'

        default_config.save_automatic = False   # if true it saves in the db_increment, otherwise a number of runs can be given
        default_config.save_observations = True
        default_config.keep_saved_runs_in_memory = True
        default_config.keep_saved_observations_in_memory = True

        default_config.memory_size_run_data = None  # number of runs that are kept in memory: 'infinity', None - no limit, int - number of capacity, 'db_increments'
        default_config.load_run_data_incremental = False
        default_config.load_observations = True # should observations automatically loaded?
        default_config.memory_size_observations = None # in case that observations should be loaded
        default_config.load_observations_incremental = False

        return default_config


    def get_memory_size_observations(self):

        memory_size_observations = self.config.memory_size_observations

        if memory_size_observations is None or memory_size_observations == 'infinity':
            memory_size_observations = None
        elif memory_size_observations == 'incremental':
            memory_size_observations = self.db_observations_increments

        return memory_size_observations


    def get_memory_size_run_data(self):

        memory_size_run_data = self.config.memory_size_run_data

        if memory_size_run_data is None or memory_size_run_data == 'infinity':
            memory_size_run_data = None
        elif memory_size_run_data == 'incremental':
            memory_size_run_data = self.db_run_data_increments

        return memory_size_run_data


    def __init__(self, config=None, **kwargs):

        self.config = ad.config.set_default_config(kwargs, config, self.__class__.default_config())

        if self.__class__.__name__ == 'ExplorationDataHandler' and self.config.db_type != 'memory':
            raise NotImplementedError('The constructor of the ExplorationDataHandler class can not be used directly if the db_type is not \'memory\'! Please use its static \'create\' method instead.')

        self.data = DataEntry()
        self.data.runs = OrderedDict()

        self.automatic_save_events = []

        self.unsaved_run_ids = set()
        self.is_exploration_data_unsaved = False

        self.run_data_ids_in_memory = [] # list of run_ids that are hold in memory
        self.observation_ids_in_memory = []  # list of run_ids that are hold in memory

        self.db_run_data_increments = 1
        self.db_observations_increments = 1

        self.run_ids = set()  # list with run_ids that exist in the db


    @classmethod
    def create(cls, config=None, **kwargs):

        db_type = 'SingleFileDB'

        if config is not None and 'db_type' in config:
            db_type = config['db_type']

        if 'db_type' in kwargs:
            db_type = kwargs['db_type']

        if db_type.lower() == 'singlefiledb' or db_type.lower() == 'file_repetition':
            return SingleFileDB(config, **kwargs)
        elif db_type.lower() == 'memory':
            return ExplorationDataHandler(config, **kwargs)
        else:
            raise ValueError('Unknown config.db_type {!r}!'.format(db_type))


    def register_automatic_save_event(self, function):
        self.automatic_save_events.append(function)


    def __getitem__(self, key):
        '''Allows to use the datahandler similar to a dict.'''
        if isinstance(key, int):
            try:
                return self.data.runs[key]
            except KeyError:
                # if run does not exists, look if it is in the db, but not in memory:
                if key not in self.run_ids:
                    raise KeyError('Run with ID {!r} does not exists in the database!')
                else:
                    # load the run_data and add it to the memory
                    self.load_run_data(ids=key)
                    return self.data.runs[key]
        else:
            return self.data[key]


    def __getattr__(self, key):
        '''Allows to use the datahandler similar to a dict.'''
        return getattr(self.data, key)


    def __dir__(self):
        return self.data.__dir__()


    def __iter__(self):
        #ids = sorted(list(self.data.runs.keys()))
        ids = sorted(self.run_ids)
        for id in ids:
            yield self[id]


    def __len__(self):
        #return self.data.runs.__len__()
        return self.run_ids.__len__()


    def __contains__(self, item):
        if isinstance(item, int):
            #return self.data.runs.__contains__(item)
            return self.run_ids.__contains__(item)
        else:
            return self.data.__contains__(item)


    def add_exploration_data(self, exploration_data=None, **kwargs):

        if exploration_data is not None:
            for argname, argvalue in exploration_data.items():
                self.data[argname] = argvalue

        if kwargs:
            for argname, argvalue in kwargs.items():
                self.data[argname] = argvalue

        self.is_exploration_data_unsaved = True

        self.automatic_save()


    def add_run_data(self, id, run_data=None, **kwargs):

        #if id not in self.data.runs:
        if id not in self.run_ids:

            if run_data is None:
                run_data = {}

            run_data_entry = RunDataEntry(self, {**kwargs, **run_data})
            run_data_entry.id = id

            self.add_run_data_to_memory(id, run_data_entry)

            self.run_ids.add(id)

        else:
            data = self.data.runs[id]

            if run_data is not None:
                for argname, argvalue in run_data.items():
                    data[argname] = argvalue

            if kwargs:
                for argname, argvalue in kwargs.items():
                    data[argname] = argvalue

        self.unsaved_run_ids.add(id)

        self.automatic_save()


    def load_run_data(self, ids, verbose=False):
        '''Loads the data for a list of runs and adds them to the memory.'''

        # Load the run_data
        run_data_list = self.load_run_data_from_db(run_ids=ids, load_incremental=self.config.load_run_data_incremental, verbose=verbose)

        # Add data to the memory
        # n_run_data = 0
        for run_id, run_data in run_data_list.items():

            # if n_run_data >= self.config.memory_size_run_data:
            #     warnings.warn('More run_data elements loaded ({}) than allowed in \'config.memory_size_run_data = {}\'. Not all element are loaded into the memory.'.format(len(run_data_list), self.config.memory_size_run_data))
            #     break

            self.add_run_data_to_memory(run_id, run_data)

            # n_run_data += 1

        return run_data_list


    def load_observations(self, ids, verbose=False):
        '''Loads the observations for the given run ids and adds them to the memory.'''

        # load observations
        observations_list = self.load_observations_from_db(run_ids=ids, load_incremental=self.config.load_observations_incremental, verbose=verbose)

        # add observations to the memory
        # n_observations = 0
        for run_id, observations in observations_list.items():
            #
            # if n_observations >= self.config.memory_size_observations:
            #     warnings.warn('More observations loaded ({}) than allowed in \'config.memory_size_observations = {}\'. Not all observations are loaded into the memory.'.format(len(observations_list), self.config.memory_size_observations))
            #     break

            self.add_observations_to_memory(run_id, observations)
            #
            # n_observations += 1

        return observations_list


    def add_run_data_to_memory(self, id, run_data):

        try:
            elem_idx = self.run_data_ids_in_memory.index(id)

            # run is already in the memory: push its idx to the top of the list
            del(self.run_data_ids_in_memory[elem_idx])
            self.run_data_ids_in_memory.insert(0, id)

        except ValueError:
            # not in the list: add it to it at the beginning and delete the last element from the memory if there are too many

            self.data.runs[id] = run_data
            self.run_data_ids_in_memory.insert(0, id)

            memory_size_run_data = self.config.memory_size_run_data
            if memory_size_run_data != 'infinity' and memory_size_run_data is not None:

                if memory_size_run_data == 'incremental':
                    memory_size_run_data = self.db_run_data_increments

                if len(self.run_data_ids_in_memory) > memory_size_run_data:

                    # take extra care if an unsaved element should be erased from memory
                    if self.run_data_ids_in_memory[-1] in self.unsaved_run_ids:
                        if self.config.save_automatic is False:
                            warnings.warn('Memory limit ({}) for run entries are reached. Because automatic saving is off (\'config.save_automatic\') the memory_limit is ignored!'.format(memory_size_run_data))
                            return
                        else:
                            # memory limit reached and automatic save is on: Save automatic
                            self.do_automatic_save()

                    # remove last element from memory

                    # if run_data is earesed, also the observation is erased
                    try:
                        self.observation_ids_in_memory.remove(self.run_data_ids_in_memory[-1])
                    except ValueError:
                        pass

                    del(self.data.runs[self.run_data_ids_in_memory[-1]])
                    del(self.run_data_ids_in_memory[-1])


        # add observations in memory if they are present
        if run_data.is_observations_loaded():
            self.add_observations_to_memory(id, run_data.observations)


    def add_observations_to_memory(self, id, observations):

        if id in self.observation_ids_in_memory:
            # observation exists already:

            self.data.runs[id].observations = observations

            # push id to start of the list
            self.observation_ids_in_memory.remove(id)
            self.observation_ids_in_memory.insert(0, id)

        else:
            # observation has to be added

            self.data.runs[id].observations = observations
            self.observation_ids_in_memory.insert(0, id)

            memory_size_observations = self.config.memory_size_observations
            if memory_size_observations != 'infinity' and memory_size_observations is not None:

                if memory_size_observations == 'incremental':
                    memory_size_observations = self.db_observations_increments

                if len(self.observation_ids_in_memory) > memory_size_observations:

                    if self.config.db_type.lower() == 'memory':
                        warnings.warn('Memory limit ({}) for observations are reached. Older observations have been deleted.'.format(memory_size_observations))
                    else:

                        # take extra care if an unsaved element should be erased from memory
                        if self.observation_ids_in_memory[-1] in self.unsaved_run_ids:
                            if self.config.save_automatic is False:
                                warnings.warn('Memory limit ({}) for observations are reached. Because automatic saving is off (\'config.save_automatic\') the memory_limit is ignored!'.format(memory_size_observations))
                                return
                            else:
                                # memory limit reached and automatic save is on: Save automatic
                                self.do_automatic_save()

                    # remove last element from memory
                    self.data.runs[self.observation_ids_in_memory[-1]].observations = None
                    del(self.observation_ids_in_memory[-1])


    def save(self):
        '''Saves unsaved data into the database, if the db_type is not 'memory'. '''

        if self.is_exploration_data_unsaved:
            self.save_exploration_data_to_db()
            self.is_exploration_data_unsaved = False

        if self.unsaved_run_ids:

            self.save_run_data_to_db(self.unsaved_run_ids)

            if self.config.save_observations:
                self.save_observations_to_db(self.unsaved_run_ids)

            if not self.config.keep_saved_runs_in_memory:
                for run_id in self.unsaved_run_ids:
                    del self.data.runs[run_id]
                self.run_data_ids_in_memory = []
                self.observation_ids_in_memory = []

            elif not self.config.keep_saved_observations_in_memory:
                for run_id in self.unsaved_run_ids:
                    self.data.runs[run_id].observations = None
                self.observation_ids_in_memory = []

        self.unsaved_run_ids.clear()


    def automatic_save(self):
        '''Handle automatic saves'''

        if self.config.save_automatic > 0:

            # detect number of run_data after which the data should be saved

            if self.config.save_automatic is True:
                increment_size = int(self.db_run_data_increments)

                if self.config.save_observations:
                    increment_size = min(increment_size, int(self.db_observations_increments))
            else:
                increment_size = int(self.config.save_automatic)


            if len(self.unsaved_run_ids) >= increment_size:
                self.do_automatic_save()


    def do_automatic_save(self):

        # call functions for the event of automatic saving
        for event_func in self.automatic_save_events:
            event_func(self, self.unsaved_run_ids)

        self.save()


    def load(self, run_ids=None, load_observations=None, verbose=False):
        '''Loads the data base.

        :param run_ids  IDs of runs for which the data should be loaded into the memory.
                        If None is given, all ids are loaded. If an empty list is given then no data is loaded.
        '''

        if run_ids is not None and not isinstance(run_ids, list):
            run_ids = [run_ids]

        if load_observations is None:
            load_observations = self.config.load_observations

        # exploration data
        self.data = self.load_exploration_data_from_db(verbose=verbose)
        self.data.runs = OrderedDict()

        # load all run_ids from the database
        self.run_ids = self.load_run_ids_from_db()

        if run_ids is None:
            run_ids = self.run_ids

        if run_ids:

            memory_size_run_data = self.get_memory_size_run_data()

            if memory_size_run_data is not None and len(run_ids) > memory_size_run_data:
                # only load the maximum number of run_data into the memory
                run_ids = list(run_ids)[0:memory_size_run_data]

            self.load_run_data(run_ids, verbose=verbose)

            if load_observations:
                self.load_observations(run_ids, verbose=verbose)


    def filter(self, condition):
        'Provdies a filter for the data given the condition.'
        return ad.helper.misc.do_filter_boolean(self, condition)


    def load_run_ids_from_db(self):
        pass


    def load_exploration_data_from_db(self, verbose):
        pass


    def load_run_data_from_db(self, run_ids, load_incremental, verbose):
        pass


    def load_observations_from_db(self, run_ids, load_incremental, verbose):
        pass


    def save_exploration_data_to_db(self):
        pass


    def save_run_data_to_db(self, run_ids):
        pass


    def save_observations_to_db(self, run_ids):
        pass




class SingleFileDB(ExplorationDataHandler):

    @staticmethod
    def default_config():
        default_config = ExplorationDataHandler.default_config()
        default_config.directory = None
        default_config.save_compressed = True
        return default_config


    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

        if self.config.directory is None:
            raise ValueError('\'config.directory\' has to be defined for the SingleFileDB ExplorationDataHandler!')

        self.db_run_data_increment = 1
        self.db_observations_increment = 1


    def load_run_ids_from_db(self):

        run_ids = set()

        run_ids.update(SingleFileDB._find_all_exploration_ids_in_directory(self.config.directory, 'run_*_observations*'))
        run_ids.update(SingleFileDB._find_all_exploration_ids_in_directory(self.config.directory, 'run_*_statistics*'))
        run_ids.update(SingleFileDB._find_all_exploration_ids_in_directory(self.config.directory, 'run_*_data*'))

        return run_ids


    @staticmethod
    def _find_all_exploration_ids_in_directory(directory, filename_template):

        ids = set()

        filepath = os.path.join(directory, filename_template)
        file_matches = glob(filepath)

        for match in file_matches:
            id_as_str = re.findall('_(\d+).', match)
            if len(id_as_str) > 0:
                ids.add(int(id_as_str[-1])) # use the last find, because ther could be more number in the filepath, such as in a directory name

        return ids


    def load_exploration_data_from_db(self, verbose=False):

        if not os.path.exists(self.config.directory):
            raise Exception('The directory {!r} does not exits! Cannot load data.'.format(self.config.directory))

        # load general experiment data
        filename = 'exploration_data.json'
        filepath = os.path.join(self.config.directory, filename)

        try:
            with open(filepath) as f:
                exploration_data = DataEntry(json.load(f, object_hook=ad.helper.data.json_numpy_object_hook))
        except FileNotFoundError:
            exploration_data = DataEntry()

        # load experiment statistics:
        filename = 'exploration_statistics.npz'
        filepath = os.path.join(self.config.directory, filename)

        try:
            exploration_data.statistics = ad.helper.data.AttrDict(dict(np.load(filepath)))

            # numpy encapsulates scalars as darrays with an empty shape
            # recover the original type
            for stat_name, stat_val in exploration_data.statistics.items():
                if len(stat_val.shape) == 0:
                    exploration_data.statistics[stat_name] = stat_val.dtype.type(stat_val)
        except FileNotFoundError:
            pass

        return exploration_data


    def load_observations_from_db(self, run_ids, load_incremental=False, verbose=False):
        '''Loads the observations for the given run_ids'''

        # if not os.path.exists(self.config.directory):
        #     raise Exception('The directory {!r} does not exits! Cannot load data.'.format(self.config.directory))

        if run_ids is None:
            run_ids = self.run_ids
        elif isinstance(run_ids, int):
            run_ids = {run_ids}
        else:
            run_ids = set(run_ids)

        if verbose:
            progress = 0
            # variable to track progess for the progress bar
            ad.gui.print_progress_bar(iteration=progress,
                                      total=len(run_ids),
                                      prefix='Loading Observations: ')

        # load observations
        observations = OrderedDict()
        for run_id in run_ids:

            filename = 'run_{:07d}_observations.npz'.format(run_id)
            filepath = os.path.join(self.config.directory, filename)

            try:
                observations[run_id] = ad.helper.data.AttrDict(dict(np.load(filepath)))
            except FileNotFoundError:
                observations[run_id] = None

            if verbose:
                # show a progress bar
                ad.gui.print_progress_bar(iteration=progress + 1,
                                          total=len(run_ids),
                                          prefix='Loading Observations: ')
                if progress + 1 == len(run_ids):
                    print('')

        return observations


    def load_run_data_from_db(self, run_ids, load_incremental=False, verbose=False):
        '''Loads the data for the given run IDs, but not the observations'''

        if run_ids is None:
            run_ids = self.run_ids
        elif isinstance(run_ids, int):
            run_ids = {run_ids}
        else:
            run_ids = set(run_ids)

        if not os.path.exists(self.config.directory):
            raise Exception('The directory {!r} does not exits! Cannot load data.'.format(self.config.directory))

        # if no ids are given, then find all ids of the existing data
        if not run_ids:
            run_ids = self.run_ids

        if verbose:
            progress = 0
            # variable to track progess for the progress bar
            ad.gui.print_progress_bar(iteration=progress,
                                      total=len(run_ids),
                                      prefix='Loading Data: ')

        # load runs
        runs = OrderedDict()
        for run_id in run_ids:

            # load general data
            filename = 'run_{:07d}_data.json'.format(run_id)
            filepath = os.path.join(self.config.directory, filename)

            try:
                with open(filepath) as f:
                    run_data = RunDataEntry(self, json.load(f, object_hook=ad.helper.data.json_numpy_object_hook))
            except FileNotFoundError:
                run_data = RunDataEntry(datahandler=self)

            # load statistics
            filename = 'run_{:07d}_statistics.npz'.format(run_id)
            filepath = os.path.join(self.config.directory, filename)

            try:
                run_data.statistics = ad.helper.data.AttrDict(dict(np.load(filepath)))

                # numpy encapsulates scalars as darrays with an empty shape
                # recover the original type
                for stat_name, stat_val in run_data.statistics.items():
                    if len(stat_val.shape) == 0:
                        run_data.statistics[stat_name] = stat_val.dtype.type(stat_val)
            except FileNotFoundError:
                pass

            # does not load observations
            run_data.observations = None

            if verbose:
                # show a progress bar
                ad.gui.print_progress_bar(iteration=progress + 1,
                                          total=len(run_ids),
                                          prefix='Loading Data: ')
                if progress + 1 == len(run_ids):
                    print('')

                progress = progress + 1

            runs[run_id] = run_data

        return runs


    def save_exploration_data_to_db(self):

        if not os.path.exists(self.config.directory):
            os.makedirs(self.config.directory)

        save_dict = dict()
        for data_name, data_value in self.data.items():

            if data_name not in ['runs', 'statistics']:
                save_dict[data_name] = data_value

            filename = 'exploration_data.json'
            filepath = os.path.join(self.config.directory, filename)

            with open(filepath, 'w') as outfile:
                json.dump(save_dict, outfile, cls=ad.helper.data.JSONNumpyEncoder)

        # save statistics with numpy
        if 'statistics' in self.data:

            filename = 'exploration_statistics'
            filepath = os.path.join(self.config.directory, filename)

            # save results as numpy array
            if self.config.save_compressed:
                np.savez_compressed(filepath, **self.data.statistics)
            else:
                np.savez(filepath, **self.data.statistics)


    def save_run_data_to_db(self, run_ids):

        for run_id in run_ids:

            run_data = self.data.runs[run_id]

            # add all data besides the observations and statistics in a json file
            save_dict = dict()
            for data_name, data_value in run_data.items():

                if data_name not in ['statistics', 'observations', 'datahandler']:
                    save_dict[data_name] = data_value

            filename = 'run_{:07d}_data.json'.format(run_id)
            filepath = os.path.join(self.config.directory, filename)

            with open(filepath, 'w') as outfile:
                json.dump(save_dict, outfile, cls=ad.helper.data.JSONNumpyEncoder)


            # save statistics
            if 'statistics' in run_data:
                filename = 'run_{:07d}_statistics'.format(run_id)
                filepath = os.path.join(self.config.directory, filename)

                if self.config.save_compressed:
                    np.savez_compressed(filepath, **run_data.statistics)
                else:
                    np.savez(filepath, **run_data.statistics)


    def save_observations_to_db(self, run_ids):

        for run_id in run_ids:

            run_data = self.data.runs[run_id]

            # save observations if wanted
            if 'observations' in run_data:
                filename = 'run_{:07d}_observations'.format(run_id)
                filepath = os.path.join(self.config.directory, filename)

                if self.config.save_compressed:
                    np.savez_compressed(filepath, **run_data.observations)
                else:
                    np.savez(filepath, **run_data.observations)

#
#
# class CombinedRunFilesDB(ExplorationDataHandler):
#
#     @staticmethod
#     def default_config():
#         default_config = ExplorationDataHandler.default_config()
#         default_config.directory = None
#         default_config.save_compressed = True
#         return default_config
#
#
#     def __init__(self, config=None, **kwargs):
#         super().__init__(config=config, **kwargs)
#
#         if self.config.directory is None:
#             raise ValueError('\'config.directory\' has to be defined for the SingleFileDB ExplorationDataHandler!')
#
#         self.db_run_data_increment = 1000
#         self.db_observations_increment = 100
#         self.run_ids_info = None
#         # holds location of the run_ids in the database and where their data is stored
#
#
#     def get_run_ids_info(self):
#
#         if self.run_ids_info is None:
#             self.load_run_ids_from_db()
#
#         return self.run_ids_info
#
#
#     def load_run_ids_from_db(self):
#         # load the file that contains the saved run_ids and their location in the db files
#
#         filepath = os.path.join(self.config.directory, 'run_ids.npz')
#
#         try:
#             # try to load the file
#             loaded_run_ids_info = ad.helper.data.AttrDict(dict(np.load(filepath)))
#
#             self.run_ids_info = ad.helper.data.AttrDict()
#             self.run_ids_info.run_ids = loaded_run_ids_info.run_ids.tolist()
#             self.run_ids_info.run_data_file_idx = loaded_run_ids_info.run_data_file_idx.tolist()
#             self.run_ids_info.run_data_infile_idx = loaded_run_ids_info.run_data_infile_idx.tolist()
#             self.run_ids_info.observations_file_idx = loaded_run_ids_info.observations_file_idx.tolist()
#             self.run_ids_info.observations_infile_idx = loaded_run_ids_info.observations_infile_idx.tolist()
#
#         except FileNotFoundError:
#             # if file not exists then create an empty shell
#             self.run_ids_info = ad.helper.data.AttrDict()
#             self.run_ids_info.run_ids = []
#             self.run_ids_info.run_data_file_idx = []
#             self.run_ids_info.run_data_infile_idx = []
#             self.run_ids_info.observations_file_idx = []
#             self.run_ids_info.observations_infile_idx = []
#
#         return self.run_ids_info.run_ids
#
#
#     def load_exploration_data_from_db(self, verbose=False):
#
#         if not os.path.exists(self.config.directory):
#             raise Exception('The directory {!r} does not exits! Cannot load data.'.format(self.config.directory))
#
#         if self.run_ids_info is None:
#             self.load_run_ids_from_db()
#
#         # load general experiment data
#         filename = 'exploration_data.json'
#         filepath = os.path.join(self.config.directory, filename)
#
#         try:
#             with open(filepath) as f:
#                 exploration_data = DataEntry(json.load(f, object_hook=ad.helper.data.json_numpy_object_hook))
#         except FileNotFoundError:
#             exploration_data = DataEntry()
#
#         # load experiment statistics:
#         filename = 'exploration_statistics.npz'
#         filepath = os.path.join(self.config.directory, filename)
#
#         try:
#             exploration_data.statistics = ad.helper.data.AttrDict(dict(np.load(filepath)))
#
#             # numpy encapsulates scalars as darrays with an empty shape
#             # recover the original type
#             for stat_name, stat_val in exploration_data.statistics.items():
#                 if len(stat_val.shape) == 0:
#                     exploration_data.statistics[stat_name] = stat_val.dtype.type(stat_val)
#         except FileNotFoundError:
#             pass
#
#         return exploration_data
#
#
#     def load_observations_from_db(self, run_ids, load_incremental=False, verbose=False):
#         '''Loads the observations for the given run_ids'''
#
#         # if not os.path.exists(self.config.directory):
#         #     raise Exception('The directory {!r} does not exits! Cannot load data.'.format(self.config.directory))
#
#         if run_ids is None:
#             run_ids = self.run_ids
#         elif isinstance(run_ids, int):
#             run_ids = {run_ids}
#         else:
#             run_ids = set(run_ids)
#
#         if verbose:
#             progress = 0
#             # variable to track progess for the progress bar
#             ad.gui.print_progress_bar(iteration=progress,
#                                       total=len(run_ids),
#                                       prefix='Loading Observations: ')
#
#         # load observations
#         observations = OrderedDict()
#         for run_id in run_ids:
#
#             filename = 'run_{:07d}_observations.npz'.format(run_id)
#             filepath = os.path.join(self.config.directory, filename)
#
#             try:
#                 observations[run_id] = ad.helper.data.AttrDict(dict(np.load(filepath)))
#             except FileNotFoundError:
#                 observations[run_id] = None
#
#             if verbose:
#                 # show a progress bar
#                 ad.gui.print_progress_bar(iteration=progress + 1,
#                                           total=len(run_ids),
#                                           prefix='Loading Observations: ')
#                 if progress + 1 == len(run_ids):
#                     print('')
#
#         return observations
#
#
#     def load_run_data_from_db(self, run_ids, load_incremental=False, verbose=False):
#         '''Loads the data for the given run IDs, but not the observations'''
#
#         if run_ids is None:
#             run_ids = self.run_ids
#         elif isinstance(run_ids, int):
#             run_ids = {run_ids}
#         else:
#             run_ids = set(run_ids)
#
#         if not os.path.exists(self.config.directory):
#             raise Exception('The directory {!r} does not exits! Cannot load data.'.format(self.config.directory))
#
#         # if no ids are given, then find all ids of the existing data
#         if not run_ids:
#             run_ids = self.run_ids
#
#         if verbose:
#             progress = 0
#             # variable to track progess for the progress bar
#             ad.gui.print_progress_bar(iteration=progress,
#                                       total=len(run_ids),
#                                       prefix='Loading Data: ')
#
#         # load runs
#         runs = OrderedDict()
#         for run_id in run_ids:
#
#             # load general data
#             filename = 'run_{:07d}_data.json'.format(run_id)
#             filepath = os.path.join(self.config.directory, filename)
#
#             try:
#                 with open(filepath) as f:
#                     run_data = RunDataEntry(self, json.load(f, object_hook=ad.helper.data.json_numpy_object_hook))
#             except FileNotFoundError:
#                 run_data = RunDataEntry(datahandler=self)
#
#             # load statistics
#             filename = 'run_{:07d}_statistics.npz'.format(run_id)
#             filepath = os.path.join(self.config.directory, filename)
#
#             try:
#                 run_data.statistics = ad.helper.data.AttrDict(dict(np.load(filepath)))
#
#                 # numpy encapsulates scalars as darrays with an empty shape
#                 # recover the original type
#                 for stat_name, stat_val in run_data.statistics.items():
#                     if len(stat_val.shape) == 0:
#                         run_data.statistics[stat_name] = stat_val.dtype.type(stat_val)
#             except FileNotFoundError:
#                 pass
#
#             # does not load observations
#             run_data.observations = None
#
#             if verbose:
#                 # show a progress bar
#                 ad.gui.print_progress_bar(iteration=progress + 1,
#                                           total=len(run_ids),
#                                           prefix='Loading Data: ')
#                 if progress + 1 == len(run_ids):
#                     print('')
#
#                 progress = progress + 1
#
#             runs[run_id] = run_data
#
#         return runs
#
#
#     def save_exploration_data_to_db(self):
#
#         if not os.path.exists(self.config.directory):
#             os.makedirs(self.config.directory)
#
#         save_dict = dict()
#         for data_name, data_value in self.data.items():
#
#             if data_name not in ['runs', 'statistics']:
#                 save_dict[data_name] = data_value
#
#             filename = 'exploration_data.json'
#             filepath = os.path.join(self.config.directory, filename)
#
#             with open(filepath, 'w') as outfile:
#                 json.dump(save_dict, outfile, cls=ad.helper.data.JSONNumpyEncoder)
#
#         # save statistics with numpy
#         if 'statistics' in self.data:
#
#             filename = 'exploration_statistics'
#             filepath = os.path.join(self.config.directory, filename)
#
#             # save results as numpy array
#             if self.config.save_compressed:
#                 np.savez_compressed(filepath, **self.data.statistics)
#             else:
#                 np.savez(filepath, **self.data.statistics)
#
#
#     def save_run_data_to_db(self, run_ids):
#
#         run_ids_info = self.get_run_ids_info()
#
#         for run_id in run_ids_info.run_ids:
#
#             # check if the run_id was already saved, if yes then it has to be saved again
#
#             if run_id not in self.run_ids_info.run_ids:
#                 # not saved yet, append it to the end of the saved data
#
#                 run_ids_info.run_ids.append[run_id]
#
#
#
#
#
#             else:
#             # is saved --> add it to the data
#
#             run_data = self.data.runs[run_id]
#
#             # add all data besides the observations and statistics in a json file
#             save_dict = dict()
#             for data_name, data_value in run_data.items():
#
#                 if data_name not in ['statistics', 'observations', 'datahandler']:
#                     save_dict[data_name] = data_value
#
#             filename = 'run_{:07d}_data.json'.format(run_id)
#             filepath = os.path.join(self.config.directory, filename)
#
#             with open(filepath, 'w') as outfile:
#                 json.dump(save_dict, outfile, cls=ad.helper.data.JSONNumpyEncoder)
#
#             # save statistics
#             if 'statistics' in run_data:
#                 filename = 'run_{:07d}_statistics'.format(run_id)
#                 filepath = os.path.join(self.config.directory, filename)
#
#                 if self.config.save_compressed:
#                     np.savez_compressed(filepath, **run_data.statistics)
#                 else:
#                     np.savez(filepath, **run_data.statistics)
#
#
#     def save_observations_to_db(self, run_ids):
#
#         for run_id in run_ids:
#
#             run_data = self.data.runs[run_id]
#
#             # save observations if wanted
#             if 'observations' in run_data:
#                 filename = 'run_{:07d}_observations'.format(run_id)
#                 filepath = os.path.join(self.config.directory, filename)
#
#                 if self.config.save_compressed:
#                     np.savez_compressed(filepath, **run_data.observations)
#                 else:
#                     np.savez(filepath, **run_data.observations)
