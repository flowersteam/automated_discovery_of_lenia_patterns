import os
import numpy as np
import autodisc as ad
import h5py

def load_data(experiment_directory):
    dh = ad.ExplorationDataHandler.create(directory=os.path.join(experiment_directory, 'results'))
    dh.load(load_observations=False, verbose=True)

    dh.config.save_automatic = False
    dh.config.load_observations = True
    dh.config.memory_size_observations = 1
    return dh


def collect_dataset(config):

    # load statistics of each experiment to select which patterns should be used
    experiment_statistics = []

    for experiment_directory in config.experiment_directories:
        stat = ad.gui.jupyter.misc.load_statistics(experiment_directory)
        experiment_statistics.append(stat)

    n_experiments = len(config.experiment_directories)
    n_repetitions = len(config.experiment_repetitions)

    experiment_selected_runs_per_class_idxs = []

    # compute how many elements per class are wanted
    n_data_per_class = np.zeros(len(config.classes), dtype=np.int32)
    for class_idx, class_definition in enumerate(config.classes):

        if class_definition['ratio'] is None or class_definition['ratio'] == 'remainder':

            if class_idx != len(config.classes) - 1:
                raise ValueError('Class with \'remainder\' as ratio must be the final class definition!')

            n_data_per_class[class_idx] = config.n_data - np.sum(n_data_per_class)

        else:
            n_data_per_class[class_idx] = int(class_definition['ratio'] * config.n_data)

    if np.sum(n_data_per_class) != config.n_data:
        raise ValueError('Class ratios do not sum up to 1.')

    # select which patterns should be used from the statistics, according to configuration
    for class_idx, class_definition in enumerate(config.classes):

        experiment_filtered_run_idxs = []

        n_max_data_in_experiment = np.zeros((n_experiments, n_repetitions), dtype=np.int32) # how much data is in the experiment

        for experiment_idx, experiment_statistic in enumerate(experiment_statistics):

            repetition_filtered_run_idxs = []

            for repetition_idx , repetition_id in enumerate(config.experiment_repetitions):

                filtered_run_inds = ad.gui.jupyter.misc.filter_single_experiment_data(experiment_statistic, class_definition['filter'], repetition_id)
                filtered_run_idxs = np.where(filtered_run_inds)[0]

                repetition_filtered_run_idxs.append(filtered_run_idxs)

                n_max_data_in_experiment[experiment_idx, repetition_idx] = len(filtered_run_idxs)

            experiment_filtered_run_idxs.append(repetition_filtered_run_idxs)

        n_data_from_experiment = np.zeros((n_experiments, n_repetitions), dtype=np.int32)  # how much data should be taken from the experiment
        n_remaining_data = n_data_per_class[class_idx] # remaining number of datapoints that need to be distributed over the experiments

        if np.sum(n_max_data_in_experiment) < n_data_per_class[class_idx]:
            raise ValueError('Experiments have not enough data points for class {}! {:.0f} data points instead of {:.0f}.'.format(class_idx,
                                                                                                                                  np.sum(n_max_data_in_experiment),
                                                                                                                                  n_data_per_class[class_idx]))

        # simple method to distribute datapoints over experiments.
        # loop over them and increase their number of datapoints until all are given
        experiment_idx = 0
        repetition_idx = 0
        while n_remaining_data > 0:

            # if the experiment has still enough data, take from it
            if n_data_from_experiment[experiment_idx, repetition_idx] < n_max_data_in_experiment[experiment_idx, repetition_idx]:
                n_data_from_experiment[experiment_idx, repetition_idx] += 1
                n_remaining_data -= 1

            experiment_idx = experiment_idx + 1
            if experiment_idx == n_experiments:
                experiment_idx = 0

                repetition_idx += 1
                if repetition_idx == n_repetitions:
                    repetition_idx = 0

        # select ids for each experiment that should be taken for this class
        experiment_cur_class_selected_run_idxs = []
        for experiment_idx in range(n_experiments):

            repetition_cur_class_selected_run_idxs = []
            for repetition_idx in range(n_repetitions):

                selected_idxs = []
                if len(experiment_filtered_run_idxs[experiment_idx][repetition_idx]) > 0:
                    selected_idxs = np.random.choice(experiment_filtered_run_idxs[experiment_idx][repetition_idx],
                                                     n_data_from_experiment[experiment_idx, repetition_idx])
                repetition_cur_class_selected_run_idxs.append(selected_idxs)
            experiment_cur_class_selected_run_idxs.append(repetition_cur_class_selected_run_idxs)
        experiment_selected_runs_per_class_idxs.append(experiment_cur_class_selected_run_idxs)

    # select images for different dataset parts
    n_train = int(config.ratio_train * config.n_data)
    n_valid = int(config.ratio_valid * config.n_data)
    n_test = config.n_data - n_train - n_valid

    # randomize the selected images
    images_idxs = np.arange(config.n_data)
    np.random.shuffle(images_idxs)

    train_idxs = images_idxs[: n_train]
    valid_idxs = images_idxs[n_train: n_train + n_valid]
    test_idxs = images_idxs[n_train + n_valid:]

    # create result folder if not exists
    if not os.path.exists(config.output_data_directory):
        os.makedirs(config.output_data_directory)

    # create hdf5 dataset file
    dataset_filename = os.path.join(config.output_data_directory, 'dataset.h5')
    with h5py.File(dataset_filename, "w") as dataset_file:

        # add some meta data
        dataset_file.attrs['n_data'] = config.n_data

        # create different subgroups of the dataset and their respective data arrays:
        train_grp = dataset_file.create_group('train')
        train_grp.attrs['n_data'] = n_train
        train_observations = train_grp.create_dataset('observations', (n_train, config.img_size[0], config.img_size[1]), config.img_dtype)
        train_labels = train_grp.create_dataset('labels', (n_train, ))

        valid_grp = dataset_file.create_group('valid')
        valid_grp.attrs['n_data'] = n_valid
        valid_observations = valid_grp.create_dataset('observations', (n_valid, config.img_size[0], config.img_size[1]), config.img_dtype)
        valid_labels = valid_grp.create_dataset('labels', (n_valid, ))

        test_grp = dataset_file.create_group('test')
        test_grp.attrs['n_data'] = n_test
        test_observations = test_grp.create_dataset('observations', (n_test, config.img_size[0], config.img_size[1]), config.img_dtype)
        test_labels = test_grp.create_dataset('labels', (n_test, ))

        overall_img_idx = 0
        train_idx = 0
        test_idx = 0
        valid_idx = 0

        # load each experiment and collect its data and write in dataset
        for experiment_idx, experiment_directory in enumerate(config.experiment_directories):

            for repetition_idx, repetition_id in enumerate(config.experiment_repetitions):

                # load the data
                experiment_repetition_directory = os.path.join(experiment_directory, 'repetition_{:06d}'.format(repetition_id))
                datahandler = load_data(experiment_repetition_directory)

                # go through data one by one per class
                for class_idx in range(len(config.classes)):

                    run_idxs = experiment_selected_runs_per_class_idxs[class_idx][experiment_idx][repetition_idx]

                    for run_idx in run_idxs:

                        run_data = datahandler[int(run_idx)]

                        obs = run_data.observations
                        final_observation = obs['states'][config.timepoint]

                        if overall_img_idx in test_idxs:
                            test_observations[test_idx, :] = final_observation
                            test_labels[test_idx] = class_idx
                            test_idx += 1
                        elif overall_img_idx in valid_idxs:
                            valid_observations[valid_idx, :] = final_observation
                            valid_labels[valid_idx] = class_idx
                            valid_idx += 1
                        else:
                            train_observations[train_idx, :] = final_observation
                            train_labels[train_idx] = class_idx
                            train_idx += 1

                        overall_img_idx += 1

    with open(os.path.join(config.output_data_directory, 'dataset_summary.csv'), 'w') as f:
        f.write('n_tot\t{}\n'.format(config.n_data))
        f.write('n_train\t{}\n'.format(n_train))
        f.write('n_valid\t{}\n'.format(n_valid))
        f.write('n_test\t{}\n'.format(n_test))
        f.write('train_ids\t{}\n'.format(train_idxs))
        f.write('valid_ids\t{}\n'.format(valid_idxs))
        f.write('test_ids\t{}\n'.format(test_idxs))


if __name__ == '__main__':

    config = ad.Config()

    config.experiment_directories = [
        '../../explorations/experiments/experiment_000001/',
        '../../explorations/experiments/experiment_000002/',
        '../../explorations/experiments/experiment_000101/',
        '../../explorations/experiments/experiment_000102/',
        '../../explorations/experiments/experiment_000103/',
        '../../explorations/experiments/experiment_000104/',
        '../../explorations/experiments/experiment_000105/',
        '../../explorations/experiments/experiment_000106/',
        '../../explorations/experiments/experiment_000107/',
        '../../explorations/experiments/experiment_000108/',
        '../../explorations/experiments/experiment_000109/',
        '../../explorations/experiments/experiment_000201/',
        '../../explorations/experiments/experiment_000202/',
        '../../explorations/experiments/experiment_000203/',
        '../../explorations/experiments/experiment_000301/',
        '../../explorations/experiments/experiment_000302/',
        '../../explorations/experiments/experiment_000303/',
    ]

    config.experiment_repetitions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    config.n_data = 50000

    config.classes = [
        dict(ratio = 1.0, filter = (('classifier_animal.data', '==', True), 'or', ('classifier_animal.data', '==', False)))  # all runs
    ]

    config.ratio_train = 0.75
    config.ratio_valid = 0.1

    config.output_data_directory = './dataset'

    config.img_size = (256, 256)
    config.img_dtype = np.float64

    config.timepoint = 0

    collect_dataset(config)
