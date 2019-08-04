import numpy as np
import os
import glob
import zipfile

def calc_experiment_statistics(statistics, load_experiment_data_func,  *args, statistics_directory='statistics', recalculate_statistics=False, verbose=False):
    '''

    :param statistics: List with tuples of the form: (statistic name, statistic function)
    :param args: Directoryies in which the experiments are for which the statistics should be computed.
    :param results_directory:
    :param statistics_directory:
    :return:
    '''

    if len(args) == 0:
        experiments = ['.']
    elif len(args) == 1 and isinstance(args[0], list):
        experiments = args[0]
    else:
        experiments = list(args)

    def get_data(data, experiment_folder):
        '''Loads the data if the data is not already loaded'''
        if data is None:
            data = load_experiment_data_func(experiment_folder)
        return data


    # identify the experiment folders
    experiment_folders = []
    for folder in experiments:

        found_folders = []

        # make sure the folder has folder form, i.e. with final '/'
        # then take the base directory and search in it
        # this makes sure, that also the direct given folder is looked after, e.g. './experiment_000001' would also be identified
        basedir = os.path.dirname(os.path.join(folder, ''))
        directory_name = os.path.basename(os.path.dirname(basedir))

        # look if directory itself is a repetition experiment
        if directory_name.find('repetition_') >=0 and os.path.isdir(basedir):
            found_folders.append(basedir)

        # look if there are repetition folders somewhere inside the directory
        if not found_folders:
            for foldername in glob.iglob(os.path.join(basedir,'**','repetition_*'), recursive=True):
                if os.path.isdir(foldername):
                    found_folders.append(foldername)

        # if there are no repetitions, then search for experiment folder
        if not found_folders:
            if directory_name.find('experiment_') >= 0 and os.path.isdir(basedir):
                found_folders.append(basedir)

        if not found_folders:
            for foldername in glob.iglob(os.path.join(basedir, '**', 'experiment_*'), recursive=True):
                if os.path.isdir(foldername):
                    found_folders.append(foldername)

        if not found_folders:
            # assume the given folder is the the one
            found_folders.append(folder)

        experiment_folders.extend(found_folders)

    # calc statistic if it does not exist already
    for experiment_folder in sorted(experiment_folders):

        data = None

        directory = os.path.join(experiment_folder, statistics_directory)

        if not os.path.isdir(directory):
            os.makedirs(directory)

        if verbose:
            print('Calculate statistics for {!r}:'.format(experiment_folder))

        for statistic_definition in statistics:

            if isinstance(statistic_definition, tuple):
                statistic_name = statistic_definition[0]
                statistic_func = statistic_definition[1]
                statistic_type = statistic_definition[2] if len(statistic_definition) > 2 else 'numpy'
            elif isinstance(statistic_definition, dict):
                statistic_name = statistic_definition['name']
                statistic_func = statistic_definition['function']
                statistic_type = statistic_definition['type'] if 'type' in statistic_definition else 'numpy'
            else:
                raise ValueError('Unknown format for statistic definition {!r}!'.format(statistic_definition))

            # calculate statistics if they do not exist
            filepath_npy = os.path.join(directory, '{}.npy'.format(statistic_name))
            filepath_npz = os.path.join(directory, '{}.npz'.format(statistic_name))
            filepath_zip = os.path.join(directory, '{}.zip'.format(statistic_name))
            directory_path = os.path.join(directory, statistic_name)

            if (not os.path.isfile(filepath_npy) and not os.path.isfile(filepath_npz) and not os.path.isfile(filepath_zip) and not os.path.isdir(directory_path)) or recalculate_statistics:

                if verbose:
                    print('\t{} ...'.format(statistic_name))

                data = get_data(data, experiment_folder)

                if statistic_type == 'numpy':
                    stat = statistic_func(data)

                    if isinstance(stat, dict):
                        np.savez(filepath_npz, **stat)
                    else:
                        np.save(filepath_npy, stat)

                elif statistic_type == 'zip':

                    stat = statistic_func(data)

                    if not isinstance(stat, dict):
                        raise ValueError('Only dictionaries are accepted as data type for zip statistics!')

                    zf = zipfile.ZipFile(filepath_zip,
                                         mode='w',
                                         compression=zipfile.ZIP_DEFLATED,
                                         )
                    try:
                        for sub_stat_name, sub_stat in stat.items():
                            zf.writestr(sub_stat_name, sub_stat)

                    finally:
                        zf.close()

                elif statistic_type == 'directory':
                    if not os.path.isdir(directory_path):
                        os.mkdir(directory_path)
                    statistic_func(data, directory_path)

                else:
                    ValueError('Unknown statistic type {!r}!'.format(statistic_type))



def calc_statistics_over_repetitions(statistics, load_experiment_data_func, *args, statistics_directory='statistics', recalculate_statistics=False, verbose=False):
    '''

    :param statistics: List with tuples of the form: (statistic name, statistic function)
    :param args: Directoryies in which the experiments are for which the statistics should be computed.
    :param results_directory:
    :param statistics_directory:
    :return:
    '''

    if len(args) == 0:
        experiments = ['.']
    elif len(args) == 1 and isinstance(args[0], list):
        experiments = args[0]
    else:
        experiments = list(args)

    def get_data(data, experiment_folder):
        '''Loads the data if the data is not already loaded'''
        if data is None:
            data = load_experiment_data_func(experiment_folder)
        return data

    # identify the experiment folders, i.e. folders that have repetition folders as subfolders
    experiment_directories = dict()
    for folder in experiments:


        # make sure the folder has folder form, i.e. with final '/'
        # then take the base directory and search in it
        # this makes sure, that also the direct given folder is looked after, e.g. './experiment_000001' would also be identified
        basedir = os.path.dirname(os.path.join(folder, ''))
        
        # look if there are repetition folders somewhere inside the directory
        for repetition_directory in glob.iglob(os.path.join(basedir,'**','repetition_*'), recursive=True):
            if os.path.isdir(repetition_directory):
                # remember the parent folder, i.e. the experiment folder
                experiment_directory = os.path.dirname(repetition_directory)
                
                if experiment_directory not in experiment_directories:
                    experiment_directories[experiment_directory] = set()

                experiment_directories[experiment_directory].add(repetition_directory)
                

    # calc statistic if it does not exist already
    for experiment_directory, repetition_directories in experiment_directories.items():

        data = None

        trg_directory = os.path.join(experiment_directory, statistics_directory)

        if not os.path.isdir(trg_directory):
            os.makedirs(trg_directory)

        if verbose:
            print('Calculate statistics for {!r}:'.format(experiment_directory))

        for statistic_name, statistic_func in statistics:

            # calculate statistics if they do not exist
            filename_npy = '{}.npy'.format(statistic_name)
            filename_npz = '{}.npz'.format(statistic_name)

            filepath_npy = os.path.join(trg_directory, filename_npy)
            filepath_npz = os.path.join(trg_directory, filename_npz)
            if (not os.path.isfile(filepath_npy) and not os.path.isfile(filepath_npz)) or recalculate_statistics:

                if verbose:
                    print('\t{} ...'.format(statistic_name))

                data = get_data(data, repetition_directories)

                stat = statistic_func(data)

                if isinstance(stat, dict):
                    np.savez(filepath_npz, **stat)
                else:
                    np.save(filepath_npy, stat)