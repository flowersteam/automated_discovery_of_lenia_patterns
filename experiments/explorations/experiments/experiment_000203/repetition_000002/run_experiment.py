import autodisc as ad
import experiment_config


def run_exploration():
    directory = './results'

    system = ad.systems.Lenia(system_parameters=experiment_config.get_system_parameters())
    system.statistics.append(ad.systems.lenia.LeniaClassifierStatistics())
    explorer = ad.explorers.GoalSpaceExplorer(system=system, config=experiment_config.get_explorer_config())
    explorer.system.statistics.append(ad.systems.statistics.ObservationDifferenceStatistic())

    explorer.data.config.save_automatic = True
    explorer.data.config.save_observations = True
    explorer.data.config.keep_saved_runs_in_memory = True
    explorer.data.config.keep_saved_observations_in_memory = True
    if hasattr(explorer.config, 'online_training'):
        explorer.data.config.memory_size_observations = explorer.config.online_training.n_runs_between_train_steps
    else:
        explorer.data.config.memory_size_observations = 100
    explorer.data.config.directory = directory

    print('Run exploration ...')
    try:
        explorer.run(experiment_config.get_number_of_explorations())

    except Exception as e:
        print('An error occured ...')
        print('\tSave explorer and exit with error ...')
        explorer.save()
        raise e

    print('Save explorer ...')
    explorer.save()

    print('Finished.')

    return explorer

if __name__ == '__main__':
    run_exploration()
