import numbers
import numpy as np
import autodisc as ad


def sample_value(rnd=None, config=None):
    '''Samples scalar values depending on the provided properties.'''

    if rnd is None:
        rnd = np.random.RandomState()

    val = None

    if isinstance(config, numbers.Number): # works also for booleans
        val = config

    elif config is None:
        val = rnd.rand()

    elif isinstance(config, tuple):

        if config[0] == 'continuous' or config[0] == 'continous':
            val = config[1] + (rnd.rand() * (config[2] - config[1]))

        elif config[0] == 'discrete':
            val = rnd.randint(config[1], config[2] + 1)

        elif config[0] == 'function' or config[0] is 'func':
            val = config[1](rnd, *config[2:])    # call function and give the other elements in the tupel as paramters

        elif len(config) == 2:
            val = config[0] + (rnd.rand() * (config[1] - config[0]))

        else:
            raise ValueError('Unknown parameter type {!r} for sampling!', config[0])

    elif isinstance(config, list):
        val = config[rnd.randint(len(config))] # do not use choice, because it does not work with tuples

    elif isinstance(config, dict):
        if config['type'] == 'discrete':
            val = rnd.randint(config['min'], config['max'] + 1)

        elif config['type'] == 'continuous':
            val = config['min'] + (rnd.rand() * (config['max'] - config['min']))

        elif config['type'] == 'boolean':
            val = bool(rnd.randint(0,1))

        else:
            raise ValueError('Unknown parameter type {!r} for sampling!', config['type'])

    return val


def mutate_value(val, mutation_factor=1.0, rnd=None, config=None, **kwargs):

    new_val = val

    if isinstance(val, list) or isinstance(val, np.ndarray):
        for idx in range(np.shape(val)[0]):
            new_val[idx] = mutate_value(new_val[idx], mutation_factor=mutation_factor, rnd=rnd, config=config, **kwargs)
    else:

        if rnd is None:
            rnd = np.random.RandomState()

        if config is None:
            config = kwargs
        else:
            config = {**config, **kwargs}

        if config and isinstance(config, dict):

            if 'distribution' in config:
                if config['distribution'] == 'gauss' or config['distribution'] == 'gaussian' or config['distribution'] == 'normal':
                    new_val = rnd.normal(val, config['sigma'] * max(0, mutation_factor))
                else:
                    raise ValueError('Unknown parameter type {!r} for mutation!', config['type'])

            if 'min' in config:
                new_val = max(new_val, config['min'])

            if 'max' in config:
                new_val = min(new_val, config['max'])

            if 'type' in config:
                if config['type'] == 'discrete':
                    new_val = np.round(new_val)
                elif config['type'] == 'continuous':
                    pass
                else:
                    raise ValueError('Unknown parameter type {!r} for mutation!', config['type'])

    return new_val



def sample_vector(rnd=None, config=None):

    val = None

    if isinstance(config, tuple):

        vector_length = int(sample_value(rnd, config[0]))

        val = [None]*vector_length
        for idx in range(vector_length):
            val[idx] = sample_value(rnd, config[1])

    else:
        raise ValueError('Unknown config type for sampling of a vactor!')

    return np.array(val)


def sample_bubble_image(rnd=None, config=None):
    '''Samples images that have a random number of bubbles drawn on them'''

    if rnd is None:
        rnd = np.random.Random()

    def_config = dict()
    def_config['size_x'] = 100
    def_config['size_y'] = 100
    def_config['min_value'] = 0
    def_config['max_value'] = 1
    def_config['num_of_blobs'] = 1
    def_config['is_uniform_blobs'] = False
    def_config['blob_radius'] = 10
    def_config['blob_position_x'] = None # if none, then random
    def_config['blob_position_y'] = None # if none, then random
    def_config['blob_height'] = 1
    def_config['sigma'] = 'full'  # if none --> no gradient
    def_config['is_image_repeating'] = False


    config = ad.helper.data.set_dict_default_values(config, def_config)

    # sample size of image
    size_x = sample_value(rnd, config['size_x'])
    size_y = sample_value(rnd, config['size_y'])
    min_value = sample_value(rnd, config['min_value'])
    max_value = sample_value(rnd, config['max_value'])
    num_of_blobs = sample_value(rnd, config['num_of_blobs'])
    is_image_repeating = sample_value(rnd, config['is_image_repeating'])
    is_uniform_blobs = sample_value(rnd, config['is_uniform_blobs'])

    # create initial image, which is leveled at the min_value
    image = np.ones((size_y, size_x)) * min_value

    for blob_idx in range(num_of_blobs):

        # sample blob parameters. Resample for each blob if they should be non-unifom.
        if blob_idx == 1 or not is_uniform_blobs:
            blob_radius = sample_value(rnd, config['blob_radius'])
            blob_height = sample_value(rnd, config['blob_height'])

            if config['sigma'] is 'full':
                sigma = 'full'
            else:
                sigma = sample_value(rnd, config['sigma'])

        # position of the blob
        # if none then sample over the complete array
        if config['blob_position_x'] is None:
            blob_pos_x = int(sample_value(rnd, (0, size_x-1)))
        else:
            blob_pos_x = int(sample_value(rnd, config['blob_position_x']))

        if config['blob_position_y'] is None:
            blob_pos_y = int(sample_value(rnd, (0, size_y-1)))
        else:
            blob_pos_y = int(sample_value(rnd, config['blob_position_y']))


        # fill blob either completely or use a gaussian gradiant if sigma is provided
        if sigma is 'full':
            gaussian_func = lambda x: blob_height
        else:
            gaussian_func = lambda x: blob_height * np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.)))

        # draw the blob on the image by addition
        for y_pos in range(size_y):
            for x_pos in range(size_x):

                if not is_image_repeating:
                    distance_from_blob_center = np.linalg.norm([y_pos - blob_pos_y, x_pos - blob_pos_x])
                else:
                    distance_from_blob_center = ad.helper.misc.get_min_distance_on_repeating_2d_array((size_y, size_x), (y_pos, x_pos), (blob_pos_y, blob_pos_x))

                if distance_from_blob_center <= blob_radius:
                    image[y_pos][x_pos] = image[y_pos][x_pos] + gaussian_func(distance_from_blob_center)

    image = image.clip(min_value, max_value)

    return image