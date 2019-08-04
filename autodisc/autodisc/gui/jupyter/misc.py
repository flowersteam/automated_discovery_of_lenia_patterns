# Helper functions
import glob
import os
import numpy as np
import warnings
import autodisc as ad
from io import BytesIO
from PIL import Image

def create_colormap(colors, is_marker_w=True):
    MARKER_COLORS_W = [0x5F,0x5F,0x5F,0x7F,0x7F,0x7F,0xFF,0xFF,0xFF]
    MARKER_COLORS_B = [0x9F,0x9F,0x9F,0x7F,0x7F,0x7F,0x0F,0x0F,0x0F]
    nval = 253
    ncol = colors.shape[0]
    colors = np.vstack((colors, np.array([[0,0,0]])))
    v = np.repeat(range(nval), 3)  # [0 0 0 1 1 1 ... 252 252 252]
    i = np.array(list(range(3)) * nval)  # [0 1 2 0 1 2 ... 0 1 2]
    k = v / (nval-1) * (ncol-1)  # interpolate between 0 .. ncol-1
    k1 = k.astype(int)
    c1, c2 = colors[k1,i], colors[k1+1,i]
    c = (k-k1) * (c2-c1) + c1  # interpolate between c1 .. c2
    return np.rint(c / 8 * 255).astype(int).tolist() + (MARKER_COLORS_W if is_marker_w else MARKER_COLORS_B)

def transform_image_from_colormap(image, colormap):
    '''
    Function that transforms the color palette of a PIL image 
    
    input: 
        - image: the PIL image to transform
        - colormap: the desired colormap
    output: the transformed PIL image
    '''
    image_array = np.array(image)
    image_array = np.uint8(image_array.astype(float)/255.0 * 252.0)
    transformed_image = Image.fromarray(image_array)
    transformed_image.putpalette(colormap)
    
    return transformed_image

def transform_image_PIL_to_bytes(image, image_format = 'png'):
    '''
    Function that transforms a PIL image to bytes format
    
    input: 
        - image: the PIL image to transform
        - image_format: the image format 
        
    output: the image bytes
    '''
    with BytesIO() as output:
        image.save(output, image_format)
        image_bytes = output.getvalue()
        
    return image_bytes

def load_statistics(experiment_directory):
    statistics = dict()

    if not os.path.isdir(experiment_directory):
        raise ValueError('Directory {!r} does not exist!'.format(experiment_directory))

    for file in glob.glob(os.path.join(experiment_directory, 'statistics', '*.npy')):
        stat_name = os.path.splitext(os.path.basename(file))[0]
        stat_val = np.load(file)

        if len(stat_val.shape) == 0:
            stat_val = stat_val.dtype.type(stat_val)

        statistics[stat_name] = stat_val

    for file in glob.glob(os.path.join(experiment_directory, 'statistics', '*.npz')):
        stat_name = os.path.splitext(os.path.basename(file))[0]
        stat_vals = dict(np.load(file))

        # numpy encapsulates scalars as darrays with an empty shape
        # recover the original type
        for substat_name, substat_val in stat_vals.items():
            if len(substat_val.shape) == 0:
                stat_vals[substat_name] = substat_val.dtype.type(substat_val)

        statistics[stat_name] = stat_vals

    return statistics


def get_repetition_ids(experiment_id, repetition_ids):
    # repetition_ids: Either scalar int with single id, list with several that are used for each experiment, or a dict with repetition ids per experiment.

    # get the selected repetitions
    cur_repetition_ids = repetition_ids

    if repetition_ids is None or repetition_ids == 'all' or repetition_ids == ['all']:
        # allows to index all repetitions, i.e. as using the operator ':'
        cur_repetition_ids = slice(None)
    else:

        if isinstance(cur_repetition_ids, dict):
            cur_repetition_ids = cur_repetition_ids[experiment_id]

        if not isinstance(cur_repetition_ids, list):
            cur_repetition_ids = [cur_repetition_ids]

    return cur_repetition_ids


def get_experiment_data(data=None, experiment_id=None, data_source=None, repetition_ids=None, data_filter=None, data_filter_inds=None):
    '''
    The datasource is a tuple which allows to spcifiy sub sources.
    Example: If "data[experimen_id] = dict(upperlevel=dict(lowerlevel=[1, 2, 3]))"
             then "datasource = ('upperlevel', 'lowerlevel')" can retrieve the data.

    A data_filter can be given according which the data gets filtered.
    See filter_single_experiment_data for the format of the filter.
    Otherwise boolen indeces can be given (data_filter_inds) to filter the data.
    Please note that when indeces are used and only a subset of repetitoin_ids are defined, then the filter inds must only have
    indeces for those repetitions and not all repetitions.
    '''

    if data_source is None:
        data_source = []
    elif not isinstance(data_source, tuple):
        data_source = (data_source,)

    cur_repetition_ids = get_repetition_ids(experiment_id, repetition_ids)

    rep_data = data[experiment_id]

    # go through data to get to final datasource
    for data_source_elem in data_source:

        if isinstance(data_source_elem, str) and isinstance(rep_data, np.ndarray) and data_source_elem[0] == '[' and data_source_elem[-1] == ']':
            rep_data = eval('rep_data[:,:,' + data_source_elem[1:-1] + ']')
        else:
            rep_data = rep_data[data_source_elem]

    if not isinstance(cur_repetition_ids, slice) and np.max(cur_repetition_ids) >= rep_data.shape[0]:
        warnings.warn('Experiment {!r} does not have all requested repetitions. Only the exisiting ones are loaded.'.format(experiment_id))
        cur_repetition_ids = [id for id in cur_repetition_ids if id < rep_data.shape[0]]

    if np.ndim(rep_data) == 1:
        rep_data = rep_data[cur_repetition_ids]
    else:
        rep_data = rep_data[cur_repetition_ids, :]

    if data_filter_inds is None and data_filter is not None and data_filter:
        # filter data according data_filter the given filter
        data_filter_inds = filter_single_experiment_data(rep_data, data_filter)

    # if there is a filter, apply it for each data individually
    if data_filter_inds is not None:
        rep_data = [rep_data[rep_idx, data_filter_inds[rep_idx]] for rep_idx in range(rep_data.shape[0])]

    return rep_data


def get_datasource_data(data=None, experiment_ids=None, data_source=None, repetition_ids=None):
    datasource_data = []

    if data is not None:

        # load data
        if experiment_ids is None:
            experiment_ids = ['all']
        elif not isinstance(experiment_ids, list):
            experiment_ids = [experiment_ids]
        if experiment_ids == ['all']:
            experiment_ids = list(data.keys())

        # handle data source --> make to list if not list
        if data_source is None:
            data_source = [None]
        elif data_source is not None and not isinstance(data_source, list):
            data_source = [data_source]

        for cur_data_source in data_source:
            cur_data_source_data = []

            for experiment_id in experiment_ids:
                cur_data_source_data.append(get_experiment_data(data=data,
                                                                experiment_id=experiment_id,
                                                                data_source=cur_data_source,
                                                                repetition_ids=repetition_ids))
            datasource_data.append(cur_data_source_data)

    return datasource_data, experiment_ids


def transform_color_str_to_tuple(colors):
    is_input_list = isinstance(colors, list)

    if not is_input_list:
        colors = [colors]

    out_color = []
    for color in colors:
        col_elements = color.replace('(', ',').replace(')', ',').split(',')
        out_color.append((col_elements[0], int(col_elements[1]), int(col_elements[2]), int(col_elements[3])))

    if is_input_list:
        return out_color
    else:
        return out_color[0]


def transform_color_tuple_to_str(colors):
    is_input_list = isinstance(colors, list)

    if not is_input_list:
        colors = [colors]

    out_color = ['{}({}, {}, {})'.format(color[0], color[1], color[2], color[3]) for color in colors]

    if is_input_list:
        return out_color
    else:
        return out_color[0]


def replace_str_from_dict(string, dictionary):
    out_string = string

    for key_name, new_str in dictionary.items():
        new_formated_str = '{}'.format(new_str)
        out_string = out_string.replace('<' + key_name + '>', new_formated_str)

    return out_string


def filter_single_experiment_data(data, filter, repetition_id=None):

    if isinstance(filter, tuple):

        if len(filter) == 3:

            bool_component_1 = filter_single_experiment_data(data, filter[0], repetition_id)
            bool_component_2 = filter_single_experiment_data(data, filter[2], repetition_id)

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

            val_component_1 = filter_single_experiment_data(data, filter[1], repetition_id)

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

        if isinstance(filter, str):
            try:
                ret_val = ad.helper.misc.get_sub_dictionary_variable(data, filter)

                if repetition_id is not None:
                    ret_val = ret_val[repetition_id]

            except KeyError:
                ret_val = filter
        else:
            ret_val = filter

    return ret_val


def filter_experiments_data(experiments_data, filter, repetition_id=None):
    filtered_data_inds = dict()

    for experiment_id, experiment_data in experiments_data.items():
        filtered_data_inds[experiment_id] = filter_single_experiment_data(experiment_data, filter, repetition_id=repetition_id)

    return filtered_data_inds
