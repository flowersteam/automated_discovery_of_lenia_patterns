import autodisc as ad
import numpy as np
import os
import json
import fractions
import warnings

def load_animals():

    source_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'leniaanimals.json')
    default_run_parameters = {'kn': 1, 'gn': 1}
    system_size = (256, 256)

    #################################################
    # load animal data from json file

    system_middle_point = (int(system_size[0] / 2), int(system_size[1] / 2))

    with open(source_file, encoding='utf-8') as file:
        animal_data = json.load(file)

    animal_configs = dict()
    animal_id = 0
    for animal_data_entry in animal_data:

        # only entries with a params field encode an animal
        if 'params' in animal_data_entry:

            animal_config = dict()

            animal_config['id'] = animal_id
            animal_config['name'] = animal_data_entry['name']

            params_run = {**default_run_parameters, **animal_data_entry['params']}

            # b can be a vector described  as string, e.g.: [0.5,2] = '1/2,2'
            params_run['b'] = [float(fractions.Fraction(st)) for st in params_run['b'].split(',')]

            # load init state from the description string
            animal_array = ad.systems.lenia.rle2arr(animal_data_entry['cells'])

            animal_height = animal_array.shape[0]
            animal_width = animal_array.shape[1]

            if animal_height > system_size[0] or animal_width > system_size[1]:
                warnings.warn('Animal {} (id = {}) is not loaded because it is too big ([{}, {}]).'.format(animal_data_entry['name'],
                                                                                                           animal_id,
                                                                                                           animal_array.shape[0],
                                                                                                           animal_array.shape[1]), Warning)

            else:
                # load the animal
                init_cond = np.zeros(system_size)

                y = int(system_middle_point[0] - animal_height / 2)
                x = int(system_middle_point[1] - animal_width / 2)

                init_cond[y:y + animal_height, x:x + animal_width] = animal_array
                params_run['init_state'] = init_cond

                animal_config['run_parameters'] = params_run

                animal_configs[animal_id] = animal_config

            animal_id = animal_id + 1

    return animal_configs


def test_lenia():

    ########################################################################
    # test if the system can be run

    lenia = ad.systems.Lenia()

    [observations, statistics] = lenia.run()

    assert observations is not None
    assert statistics is not None


    ########################################################################
    # test if the different version of lenia are behaving similar

    animal_configs = load_animals()

    system_parameters = ad.systems.Lenia.default_system_parameters()
    system_parameters.size_y = 256
    system_parameters.size_x = 256

    n_steps = 100
    animal_id = 5

    run_parameters = animal_configs[animal_id]['run_parameters']

    config = ad.systems.Lenia.default_config()
    config.version = 'pytorch_fft'
    lenia_pytorch_fft = ad.systems.Lenia(system_parameters=system_parameters, config=config)
    lenia_pytorch_fft_observation, lenia_pytorch_fft_statistics = lenia_pytorch_fft.run(run_parameters=run_parameters, stop_conditions=n_steps)

    config = ad.systems.Lenia.default_config()
    config.version = 'pytorch_conv2d'
    lenia_pytorch_conv2d = ad.systems.Lenia(system_parameters=system_parameters, config=config)
    lenia_pytorch_conv2d_observation, lenia_pytorch_conv2d_statistics = lenia_pytorch_conv2d.run(run_parameters=run_parameters, stop_conditions=n_steps)

    config = ad.systems.Lenia.default_config()
    config.version = 'reikna_fft'
    lenia_reikna_fft = ad.systems.Lenia(system_parameters=system_parameters, config=config)
    lenia_reikna_fft_observation, lenia_reikna_fft_statistics = lenia_reikna_fft.run(run_parameters=run_parameters, stop_conditions=n_steps)

    # final_obs_pytorch_fft = lenia_pytorch_fft_observation['states'][-1]
    # final_obs_pytorch_conv2d = lenia_pytorch_conv2d_observation['states'][-1]
    # final_obs_reikna_fft = lenia_reikna_fft_observation['states'][-1]
    #
    # diff_pytorch_fft_pytorch_conv2d = np.abs(final_obs_pytorch_fft - final_obs_pytorch_conv2d)
    # diff_pytorch_fft_reikna_fft = np.abs(final_obs_pytorch_fft - final_obs_reikna_fft)
    # diff_pytorch_conv2d_reikna_fft = np.abs(final_obs_pytorch_conv2d - final_obs_reikna_fft)
    #
    # print('Animal {}:'.format(animal_id))
    #
    # print(np.max(diff_pytorch_fft_pytorch_conv2d))
    # print(np.max(diff_pytorch_fft_reikna_fft))
    # print(np.max(diff_pytorch_conv2d_reikna_fft))
    #
    activation_mass_diff_pytorch_fft_pytorch_conv2d  = np.max(np.abs(lenia_pytorch_fft_statistics['activation_mass'] - lenia_pytorch_conv2d_statistics['activation_mass']))
    activation_mass_diff_pytorch_fft_reikna_fft = np.max(np.abs(lenia_pytorch_fft_statistics['activation_mass'] - lenia_reikna_fft_statistics['activation_mass']))
    activation_mass_diff_pytorch_conv2d_reikna_fft = np.max(np.abs(lenia_pytorch_conv2d_statistics['activation_mass'] - lenia_reikna_fft_statistics['activation_mass']))

    # make sure that the animal survived and is not dead
    assert lenia_pytorch_fft_statistics['activation_mass'][-1] > 0.001
    assert lenia_pytorch_fft_statistics['activation_mass'][-1] < 0.5

    # at the moment there can be strong difference on the pixel level between the versions, thus only check overall activation level
    assert activation_mass_diff_pytorch_fft_pytorch_conv2d < 0.0005
    assert activation_mass_diff_pytorch_fft_reikna_fft < 0.0005
    assert activation_mass_diff_pytorch_conv2d_reikna_fft < 0.0005


def test_lenia_statistics():

    system_parameters = ad.systems.Lenia.default_system_parameters()
    system_parameters.size_y = 10
    system_parameters.size_x = 10

    lenia = ad.systems.Lenia(system_parameters=system_parameters)

    lenia_stats = ad.systems.lenia.LeniaStatistics(lenia)

    obs = np.zeros((system_parameters.size_y, system_parameters.size_x))
    all_obs = [obs, obs, obs]

    lenia_stats.calc_after_run(system=lenia, all_obs=all_obs)

    np.testing.assert_equal(lenia_stats.data['activation_mass'], [0, 0, 0])
    np.testing.assert_equal(lenia_stats.data['activation_mass_mean'], 0)
    np.testing.assert_equal(lenia_stats.data['activation_mass_std'], 0)

    np.testing.assert_equal(lenia_stats.data['activation_volume'], [0, 0, 0])
    np.testing.assert_equal(lenia_stats.data['activation_volume_mean'], 0)
    np.testing.assert_equal(lenia_stats.data['activation_volume_std'], 0)

    np.testing.assert_equal(lenia_stats.data['activation_density'], [0, 0, 0])
    np.testing.assert_equal(lenia_stats.data['activation_density_mean'], 0)
    np.testing.assert_equal(lenia_stats.data['activation_density_std'], 0)

    np.testing.assert_equal(lenia_stats.data['activation_center_velocity'], [np.nan, 0, 0])
    np.testing.assert_equal(lenia_stats.data['activation_center_velocity_mean'], 0)
    np.testing.assert_equal(lenia_stats.data['activation_center_velocity_std'], 0)

