import autodisc as ad
import numpy as np
from autodisc.helper.sampling import sample_value
from autodisc.helper.sampling import sample_vector
from autodisc.helper.sampling import sample_bubble_image

def test_sample_value():

    num_of_test = 10
    rand = np.random.RandomState()

    # check scalars
    for i in range(num_of_test):

        val = sample_value(rand, 1,)
        assert val == 1

        val = sample_value(rand, 1.3)
        assert val == 1.3

        val = sample_value(rand, -1.3)
        assert val == -1.3


    # check sampling of continuous (min, max)
    for i in range(num_of_test):

        min_max = (0,1)
        val = sample_value(rand, min_max)
        assert val >= min_max[0] and val <= min_max[1]

        min_max = (-1,1)
        val = sample_value(rand, min_max)
        assert val >= min_max[0] and val <= min_max[1]

        min_max = (40,50)
        val = sample_value(rand, min_max)
        assert val >= min_max[0] and val <= min_max[1]


    # check sampling of continuous ('continous', min, max)
    for i in range(num_of_test):

        min_max = ('continous', 0,1)
        val = sample_value(rand, min_max)
        assert val >= min_max[1] and val <= min_max[2] and int(val) != val

        min_max = ('continous', -1,1)
        val = sample_value(rand, min_max)
        assert val >= min_max[1] and val <= min_max[2] and int(val) != val

        min_max = ('continous', 40,50)
        val = sample_value(rand, min_max)
        assert val >= min_max[1] and val <= min_max[2] and int(val) != val


    # check sampling of discrete ('discrete', min, max)
    for i in range(num_of_test):

        min_max = ('discrete', 0,1)
        val = sample_value(rand, min_max)
        assert val in [0, 1]

        min_max = ('discrete', -1,1)
        val = sample_value(rand, min_max)
        assert val in [-1, 0, 1]

        min_max = ('discrete', 40,50)
        val = sample_value(rand, min_max)
        assert val in [40,41,42,43,44,45,46,47,48,49,50]



    # check sampling of item from list
    for i in range(num_of_test):

        lst = [0,1,2,3]
        val = sample_value(rand, lst)
        assert val in lst

        lst = ['a','bb','ccc','dddd']
        val = sample_value(rand, lst)
        assert val in lst

        lst = [3, '22', (33, 34)]
        val = sample_value(rand, lst)
        assert val in lst


    for i in range(num_of_test):

        descr = dict()
        descr['type'] = 'continuous'
        descr['min'] = 0
        descr['max'] = 1
        val = sample_value(rand, descr)
        assert val >= descr['min'] and val <= descr['max'] and int(val) != val

        descr['type'] = 'continuous'
        descr['min'] = -10
        descr['max'] = 10
        val = sample_value(rand, descr)
        assert val >= descr['min'] and val <= descr['max'] and int(val) != val

        descr['type'] = 'discrete'
        descr['min'] = 0
        descr['max'] = 1
        val = sample_value(rand, descr)
        assert val in [0,1]

        descr['type'] = 'discrete'
        descr['min'] = 10
        descr['max'] = 15
        val = sample_value(rand, descr)
        assert val in [10,11,12,13,14,15]


    # userdefined function
    my_func = lambda m_rnd, a, b: a-b
    descr = ('function', my_func, 100, 50)
    val = sample_value(rand, descr)
    assert val == descr[2] - descr[3]

    my_func = lambda m_rnd, a, b: a-b
    descr = ('func', my_func, 100, 50)
    val = sample_value(rand, descr)
    assert val == descr[2] - descr[3]


def test_sample_discrete_value():

    # check if all possible discrete values can be found in the samples
    num_of_test = 10
    rand = np.random.RandomState(seed=2)

    descr = dict()
    descr['type'] = 'discrete'
    descr['min'] = 0
    descr['max'] = 2

    values = []

    for i in range(num_of_test):
        values.append(sample_value(rand, descr))

    assert 0 in values
    assert 1 in values
    assert 2 in values


def test_mutate_value():

    # gauss continuous

    val = ad.helper.sampling.mutate_value(1)
    assert val == 1

    val = ad.helper.sampling.mutate_value(1, config={'distribution': 'gauss', 'sigma': 0})
    assert val == 1

    for _ in range(100):
        val = ad.helper.sampling.mutate_value(1, config={'distribution': 'gauss', 'sigma': 1, 'min': 0.9, 'max': 1.1})
        assert val >= 0.9 and val <= 1.1

    val = ad.helper.sampling.mutate_value(1, distribution='gauss', sigma=0)
    assert val == 1

    for _ in range(100):
        val = ad.helper.sampling.mutate_value(1, distribution='gauss', sigma=1, min=0.9, max=1.1)
        assert val >= 0.9 and val <= 1.1


    # gauss discrete
    val = ad.helper.sampling.mutate_value(1.1, type='discrete')
    assert val == 1

    val = ad.helper.sampling.mutate_value(1.1, type='discrete', distribution='gauss', sigma=0)
    assert val == 1

    for _ in range(100):
        val = ad.helper.sampling.mutate_value(1, type='discrete', distribution='gauss', sigma=3, min=-0.5, max=2.5)
        assert val in [0, 1, 2]


    # vector
    val = ad.helper.sampling.mutate_value([1.1, 2.2], type='discrete')
    assert val == [1, 2]

    val = ad.helper.sampling.mutate_value(np.array([1.1, 2.2]), type='discrete')
    assert np.all(val == np.array([1, 2]))


    # mutation factor
    val = ad.helper.sampling.mutate_value(1, mutation_factor=0, config={'distribution': 'gauss', 'sigma': 1})
    assert val == 1



def test_sample_bubble_image():

    # set to true to display generated images
    is_inspect_manually = False

    if is_inspect_manually:
        gui_config = dict()
        gui_config['steps'] = [[0]]
        gui_config['pixel_size'] = 8

    rand = np.random.RandomState()

    #############################################################
    # test default
    config = None
    sampled_image = sample_bubble_image(rand, config)

    assert np.all(np.all(sampled_image >= 0)) and np.all(np.all(sampled_image <= 1))

    if is_inspect_manually:
        image_viewer = ad.gui.ObservationPreviewGUI(gui_config=gui_config, obs=[sampled_image])
        image_viewer.master.mainloop()

    #############################################################
    # test default

    config = dict()
    config['size_x'] = 100
    config['size_y'] = 100
    config['min_value'] = 0
    config['max_value'] = 1
    config['num_of_blobs'] = ('discrete', 1, 5)
    config['is_uniform_blobs'] = False
    config['blob_radius'] = 10
    config['blob_position_x'] = None # if none, then random
    config['blob_position_y'] = None # if none, then random
    config['blob_height'] = (0.3, 0.7)
    config['sigma'] = 'full'  # if none --> no gradient

    sampled_image = sample_bubble_image(rand, config)

    assert np.all(np.all(sampled_image >= 0)) and np.all(np.all(sampled_image <= 1))

    if is_inspect_manually:
        image_viewer = ad.gui.ObservationPreviewGUI(gui_config=gui_config, obs=[sampled_image])
        image_viewer.master.mainloop()

    #############################################################
    # test default

    config = dict()
    config['size_x'] = 100
    config['size_y'] = 100
    config['min_value'] = 0
    config['max_value'] = 1
    config['num_of_blobs'] = ('discrete', 1, 5)
    config['is_uniform_blobs'] = False
    config['blob_radius'] = 10
    config['blob_position_x'] = None # if none, then random
    config['blob_position_y'] = None # if none, then random
    config['blob_height'] = (0.3, 0.7)
    config['sigma'] = (0.7,10)  # if none --> no gradient

    sampled_image = sample_bubble_image(rand, config)

    assert np.all(np.all(sampled_image >= 0)) and np.all(np.all(sampled_image <= 1))

    if is_inspect_manually:
        image_viewer = ad.gui.ObservationPreviewGUI(gui_config=gui_config, obs=[sampled_image])
        image_viewer.master.mainloop()


    #############################################################
    # test default

    config = dict()
    config['size_x'] = 100
    config['size_y'] = 100
    config['min_value'] = 0
    config['max_value'] = 1
    config['num_of_blobs'] = ('discrete', 1, 5)
    config['is_uniform_blobs'] = False
    config['blob_radius'] = 100
    config['blob_position_x'] = None # if none, then random
    config['blob_position_y'] = None # if none, then random
    config['blob_height'] = (0.3, 0.7)
    config['sigma'] = (0.7,10)  # if none --> no gradient

    sampled_image = sample_bubble_image(rand, config)

    assert np.all(np.all(sampled_image >= 0)) and np.all(np.all(sampled_image <= 1))

    if is_inspect_manually:
        image_viewer = ad.gui.ObservationPreviewGUI(gui_config=gui_config, obs=[sampled_image])
        image_viewer.master.mainloop()


    #############################################################
    # test default

    config = dict()
    config['size_x'] = 100
    config['size_y'] = 100
    config['min_value'] = 0
    config['max_value'] = 1
    config['num_of_blobs'] = ('discrete', 1, 5)
    config['is_uniform_blobs'] = False
    config['blob_radius'] = 100
    config['blob_position_x'] = None # if none, then random
    config['blob_position_y'] = None # if none, then random
    config['blob_height'] = (0.3, 0.7)
    config['sigma'] = (0.7,10)  # if none --> no gradient
    config['is_image_repeating'] = True

    sampled_image = sample_bubble_image(rand, config)

    assert np.all(np.all(sampled_image >= 0)) and np.all(np.all(sampled_image <= 1))

    if is_inspect_manually:
        image_viewer = ad.gui.ObservationPreviewGUI(gui_config=gui_config, obs=[sampled_image])
        image_viewer.master.mainloop()



def test_sample_vector():

    rnd = np.random.RandomState()

    repetitions = 10

    for idx in range(repetitions):
        config = (3, (0, 2))
        vec = sample_vector(rnd, config)
        assert len(vec) == 3
        assert np.all(np.array(vec) >= 0) and np.all(np.array(vec) <= 2)


    for idx in range(repetitions):
        config = ((2,4), (-1, 2))
        vec = sample_vector(rnd, config)
        assert len(vec) >= 2 and len(vec) <= 4
        assert np.all(np.array(vec) >= -1) and np.all(np.array(vec) <= 2)


