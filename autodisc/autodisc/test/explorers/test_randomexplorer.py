import autodisc as ad


def test_randomexplorer():
    # use lenia as test system

    lenia = ad.systems.lenia.Lenia(statistics=[])

    config = ad.explorers.RandomExplorer.default_config()

    config.num_of_steps = 10

    random_init_state_config = dict()
    random_init_state_config['size_x'] = lenia.system_parameters['size_x']
    random_init_state_config['size_y'] = lenia.system_parameters['size_y']
    random_init_state_config['min_value'] = 0
    random_init_state_config['max_value'] = 1
    random_init_state_config['num_of_blobs'] = ('discrete', 1, 5)
    random_init_state_config['is_uniform_blobs'] = False
    random_init_state_config['blob_radius'] = (5,100)
    random_init_state_config['blob_position_x'] = None # if none, then random
    random_init_state_config['blob_position_y'] = None # if none, then random
    random_init_state_config['blob_height'] = (0.3, 1)
    random_init_state_config['sigma'] = (0.7,10)  # if none --> no gradient
    random_init_state_config['is_image_repeating'] = True

    config.explore_parameters['init_state'] = ('function', ad.helper.sampling.sample_bubble_image, random_init_state_config) # function, config
    config.explore_parameters['b'] = ('function', ad.helper.sampling.sample_vector, (('discrete', 1, 3), (0, 1)))  # function, config
    config.explore_parameters['R'] = {'type': 'discrete',     'min': 2, 'max':50}
    config.explore_parameters['T'] = {'type': 'continuous',   'min': 1, 'max':50}
    config.explore_parameters['m'] = {'type': 'continuous',   'min': 0, 'max':1}
    config.explore_parameters['s'] = {'type': 'continuous',   'min': 0, 'max':1}
    config.explore_parameters['kn']= {'type': 'discrete',     'min': 0, 'max':3}
    config.explore_parameters['gn']= {'type': 'discrete',     'min': 0, 'max': 2}

    explorer = ad.explorers.RandomExplorer(system=lenia, config=config)

    explorer.run(num_of_runs=2, verbose=False)

    assert len(explorer.data) == 2
    assert explorer.data[1].statistics == {}    # no statistics should have been computed



