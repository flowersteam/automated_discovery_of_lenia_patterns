import itertools
import numpy as np

def get_min_distance_on_repeating_2d_array(array_size, point_a, point_b):
    '''
    Returns the minimum distance between 2 points in an array that repeats itself.
    '''

    # we generate all possible points for p1 if we would replicate the image around itself
    # then calculate all possible distances and choose the minimal distance

    possible_y_pos = [point_a[0] - array_size[0], point_a[0], point_a[0] + array_size[0]]
    possible_x_pos = [point_a[1] - array_size[1], point_a[1], point_a[1] + array_size[1]]

    possible_points_a = list(itertools.product(possible_y_pos, possible_x_pos))

    distances = [np.linalg.norm([a[0] - point_b[0], a[1] - point_b[1]]) for a in possible_points_a]

    return np.min(distances)


def angle_of_vec_degree(vector):
    # if vector[0] >= 0:
    #     return np.arctan2(vector[0], vector[1]) * 180/np.pi
    # else:
    #     return 360+(np.arctan2(vector[0], vector[1]) * 180 / np.pi)

    if vector[1] >= 0:
        return np.arctan2(vector[1], vector[0]) * 180/np.pi
    else:
        return 360+(np.arctan2(vector[1], vector[0]) * 180 / np.pi)


def angle_difference_degree(angle1, angle2):
    '''
    Calculates the disctance between two angles which are in degree.

    If the distance is clockwise, its negative, otherwise positive.

    :param angle1: First angle. Either scalar or array.
    :param angle2: Second angle. Either scalar or array.
    :return: Distance between angles in degrees.
    '''
    if isinstance(angle1, list):
        angle1 = np.array(angle1)

    if isinstance(angle2, list):
        angle2 = np.array(angle2)

    phi = np.mod(angle2 - angle1, 360)

    if not isinstance(phi, np.ndarray):
        sign = 1
        # used to calculate sign
        if not ((phi >= 0 and phi <= 180) or (
                phi <= -180 and phi >= -360)):
            sign = -1
        if phi > 180:
            result = 360 - phi
        else:
            result = phi

    else:
        sign = np.ones(phi.shape)

        sign[np.logical_not( np.logical_or(np.logical_and(phi >= 0, phi <= 180), np.logical_and(phi <= -180, phi >= -360)) )] = -1

        result = phi
        result[phi > 180] = 360 - phi[phi > 180]

    return result * sign



def get_sub_dictionary_variable(base_dict, variable):

    var_sub_elems = variable.split('.')

    cur_elem = base_dict
    for sub_elem in var_sub_elems:

        sub_elem_string_elements = sub_elem.split('[')
        sub_elem_basename = sub_elem_string_elements[0]

        if sub_elem_basename not in cur_elem:
            raise KeyError('Subelement {!r} does not exist in the base dictionary.'.format(sub_elem_basename))

        if len(sub_elem_string_elements) == 1:
            cur_elem = cur_elem[sub_elem_basename]
        else:
            sub_elem_index = sub_elem_string_elements[1].replace(']', '')

            try:
                sub_elem_index = int(sub_elem_index)
            except Exception as err:
                raise NotImplementedError('The sub indexing does only allow single indexes!') from err

            cur_elem = cur_elem[sub_elem_basename][sub_elem_index]

    return cur_elem


def do_filter_boolean(data, filter):

    if isinstance(filter, tuple):

        if len(filter) == 3:

            bool_component_1 = do_filter_boolean(data, filter[0])
            bool_component_2 = do_filter_boolean(data, filter[2])

            if filter[1] == 'and':
                ret_val = bool_component_1 & bool_component_2
            elif filter[1] == 'or':
                ret_val = bool_component_1 | bool_component_2
            elif filter[1] == '<':
                ret_val = bool_component_1 < bool_component_2
            elif filter[1] == '<=':
                ret_val = bool_component_1 <= bool_component_2
            elif filter[1] == '>':
                ret_val = bool_component_1 > bool_component_2
            elif filter[1] == '>=':
                ret_val = bool_component_1 >= bool_component_2
            elif filter[1] == '==':
                ret_val = bool_component_1 == bool_component_2
            elif filter[1] == '!=':
                ret_val = bool_component_1 != bool_component_2
            elif filter[1] == '+':
                ret_val = bool_component_1 + bool_component_2
            elif filter[1] == '-':
                ret_val = bool_component_1 - bool_component_2
            elif filter[1] == '*':
                ret_val = bool_component_1 * bool_component_2
            elif filter[1] == '/':
                ret_val = bool_component_1 / bool_component_2
            elif filter[1] == '%':
                ret_val = bool_component_1 % bool_component_2
            else:
                raise ValueError('Unknown operator {!r}!'.format(filter[1]))

        elif len(filter) == 2:

            val_component_1 = do_filter_boolean(data, filter[1])

            if filter[0] == 'sum':
                ret_val = np.sum(val_component_1)
            elif filter[0] == 'cumsum':
                ret_val = np.cumsum(val_component_1)
            elif filter[0] == 'max':
                ret_val = np.max(val_component_1)
            elif filter[0] == 'min':
                ret_val = np.min(val_component_1)
            else:
                raise ValueError('Unknown operator {!r}!'.format(filter[0]))

        else:
            raise ValueError('Unknown filter command {!r}!'.format(filter))

    else:

        is_var = False
        if isinstance(filter, str):

            # check if string is a variable in the data
            is_var = True
            try:

                if isinstance(data, list) or isinstance(data, np.ndarray):
                    get_sub_dictionary_variable(data[0], filter)
                else:
                    # check first item if the data object has a __iter__ method such as the explorationdatahandler
                    for item in data:
                        get_sub_dictionary_variable(item, filter)
                        break

            except KeyError:
                is_var = False

        if is_var:
            # if the string is a variable then get the data of the variable:
            ret_val = np.zeros(len(data))

            for data_idx, cur_data in enumerate(data):
                ret_val[data_idx] = get_sub_dictionary_variable(cur_data, filter)
        else:
            ret_val = filter

    return ret_val