import numpy as np
from autodisc.helper.misc import get_min_distance_on_repeating_2d_array
from autodisc.helper.misc import angle_of_vec_degree
from autodisc.helper.misc import angle_difference_degree
from autodisc.helper.misc import do_filter_boolean
import autodisc as ad

def test_get_min_distance_on_repeating_2d_array():

    array_size=(10,10)
    point_a = (1,0)
    point_b = (3,0)
    dist = get_min_distance_on_repeating_2d_array(array_size, point_a, point_b)
    assert dist == 2

    array_size=(10,10)
    point_a = (1,0)
    point_b = (9,0)
    dist = get_min_distance_on_repeating_2d_array(array_size, point_a, point_b)
    assert dist == 2

    array_size=(10,10)
    point_a = (1,0)
    point_b = (8,0)
    dist = get_min_distance_on_repeating_2d_array(array_size, point_a, point_b)
    assert dist == 3


def test_angle_of_vec():

    assert 0 == angle_of_vec_degree([1, 0])
    assert 45 == angle_of_vec_degree([1, 1])
    assert 90 == angle_of_vec_degree([0, 1])
    assert 135 == angle_of_vec_degree([-1, 1])
    assert 180 == angle_of_vec_degree([-1, 0])
    assert 225 == angle_of_vec_degree([-1, -1])
    assert 270 == angle_of_vec_degree([0, -1])
    assert 315 == angle_of_vec_degree([1, -1])


def test_angle_difference_degree():

    angle1 = 0
    angle2 = 90
    angle_diff = angle_difference_degree(angle1, angle2)
    assert angle_diff == 90

    angle1 = 0
    angle2 = 180
    angle_diff = angle_difference_degree(angle1, angle2)
    assert angle_diff == 180

    angle1 = 0
    angle2 = 270
    angle_diff = angle_difference_degree(angle1, angle2)
    assert angle_diff == -90

    angle1 = 0
    angle2 = 0
    angle_diff = angle_difference_degree(angle1, angle2)
    assert angle_diff == 0

    angle1 = 0
    angle2 = 360
    angle_diff = angle_difference_degree(angle1, angle2)
    assert angle_diff == 0

    angle1 = 10
    angle2 = 350
    angle_diff = angle_difference_degree(angle1, angle2)
    assert angle_diff == -20

    angle1 = 340
    angle2 = 5
    angle_diff = angle_difference_degree(angle1, angle2)
    assert angle_diff == 25


    # matrices
    angle1 = 0
    angle2 = [[90, 180, 270],
              [0, 360, 10]]

    test_diff = np.array([[90, 180, -90],
                          [0, 0, 10]])

    angle_diff = angle_difference_degree(angle1, angle2)
    assert np.all(np.equal(angle_diff, test_diff))


def test_boolean_filtering():

    # create data to filter:
    data = [dict(x=2, y=2),
            dict(x=2, y=3),
            dict(x=3, y=2),
            ]

    filtered = do_filter_boolean(data, ('x', '>', 4))
    assert np.all(np.array([False, False, False]) == filtered)

    filtered = do_filter_boolean(data, ('y', '>=', 3))
    assert np.all(np.array([False, True, False]) == filtered)

    filtered = do_filter_boolean(data, ('x', '<', 3))
    assert np.all(np.array([True, True, False]) == filtered)

    filtered = do_filter_boolean(data, ('x', '<=', 2))
    assert np.all(np.array([True, True, False]) == filtered)

    filtered = do_filter_boolean(data, ('x', '==', 2))
    assert np.all(np.array([True, True, False]) == filtered)

    filtered = do_filter_boolean(data, ('x', '==', 'y'))
    assert np.all(np.array([True, False, False]) == filtered)

    filtered = do_filter_boolean(data, ('x', '!=', 'y'))
    assert np.all(np.array([False, True, True]) == filtered)

    filtered = do_filter_boolean(data, (('x', '<', 3) , 'and', ('y', '==', 3)))
    assert np.all(np.array([False, True, False]) == filtered)

    filtered = do_filter_boolean(data, (('x', '==', 3) , 'or', ('y', '==', 3)))
    assert np.all(np.array([False, True, True]) == filtered)

    filtered = do_filter_boolean(data, (('cumsum', 'x'), '==', 4 ))
    assert np.all(np.array([False, True, False]) == filtered)

    filtered = do_filter_boolean(data, ('sum', 'x'))
    assert np.all(7 == filtered)

    filtered = do_filter_boolean(data, ('max', 'x'))
    assert np.all(3 == filtered)

    filtered = do_filter_boolean(data, ('min', 'x'))
    assert np.all(2 == filtered)


def test_get_sub_dictionary_variable():

    test_dict = dict(var_a = 1,
                     sub_dict_1 = dict(
                         var_b = 2,
                         sub_dict_2 = dict(
                             var_c = [3, 4, 5],
                            )
                        )
                     )

    var = ad.helper.misc.get_sub_dictionary_variable(test_dict, 'var_a')
    assert var == test_dict['var_a']

    var = ad.helper.misc.get_sub_dictionary_variable(test_dict, 'sub_dict_1.var_b')
    assert var == test_dict['sub_dict_1']['var_b']

    var = ad.helper.misc.get_sub_dictionary_variable(test_dict, 'sub_dict_1.sub_dict_2.var_c')
    assert var == test_dict['sub_dict_1']['sub_dict_2']['var_c']

    var = ad.helper.misc.get_sub_dictionary_variable(test_dict, 'sub_dict_1.sub_dict_2.var_c[1]')
    assert var == test_dict['sub_dict_1']['sub_dict_2']['var_c'][1]

    var = ad.helper.misc.get_sub_dictionary_variable(test_dict, 'sub_dict_1.sub_dict_2.var_c[-1]')
    assert var == test_dict['sub_dict_1']['sub_dict_2']['var_c'][-1]
