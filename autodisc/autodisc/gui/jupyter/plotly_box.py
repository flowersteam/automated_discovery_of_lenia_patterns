import autodisc as ad
import numpy as np
import plotly

# plotly_box
def plotly_box(data=None, config=None, **kwargs):
    '''
    param repetition_ids: Either scalar int with single id, list with several that are used for each experiment, or a dict with repetition ids per experiment.
    '''
    default_config = dict(
        subplots=dict(
            rows=None,
            cols=None,
            print_grid=False
        ),
        init_mode='mean_std',  # mean_std, mean, elements
        layout=dict(

            default_xaxis=dict(),  # if several subplots, then these are the default values for all xaxis config in fig.layout
            default_yaxis=dict(),  # if several subplots, then these are the default values for all yaxis config in fig.layout

            xaxis=dict(),
            yaxis=dict(),

            boxmode='group',

            updatemenus=[
                dict(type="buttons",
                     active=0,
                     buttons=[
                         dict(label='normal',
                              method='restyle',
                              args=[{'boxpoints': False}]),
                         dict(label='all',
                              method='restyle',
                              args=[{'boxpoints': 'all'}]),
                         dict(label='suspectedoutliers',
                              method='restyle',
                              args=[{'boxpoints': 'suspectedoutliers'}]),
                         dict(label='outliers',
                              method='restyle',
                              args=[{'boxpoints': 'outliers'}]),
                     ],
                     direction='right',
                     pad={'t': 70},
                     x=1,
                     xanchor='right',
                     y=0,
                     yanchor='top'
                     ),
            ]
        ),

        default_trace=dict(
            legendgroup=None,
            boxmean='sd',
        ),
        default_subplot_traces=[],
        traces=[],

        default_trace_label='<trace_idx>',
        trace_labels=[],

        default_group_label='<group_idx>',
        group_labels=[],

        default_colors=plotly.colors.DEFAULT_PLOTLY_COLORS,
    )
    config = ad.config.set_default_config(kwargs, config, default_config)

    if data is None:
        data = np.array([])

    # format data in form [subplot_idx:list][trace_idx:list][elems_per_trace:numpy.ndarray]
    if isinstance(data, np.ndarray):
        data = [[data]]
    elif isinstance(data[0], list) and isinstance(data[0][0], np.ndarray):
        data = [data]

    # identify the number of subplots
    n_subplots = len(data)

    # if not defined, set rows and cols of subplots
    if config.subplots.rows is None and config.subplots.cols is None:
        config.subplots.rows = n_subplots
        config.subplots.cols = 1
    elif config.subplots.rows is not None and config.subplots.cols is None:
        config.subplots.cols = int(np.ceil(n_subplots / config.subplots.rows))
    elif config.subplots.rows is None and config.subplots.cols is not None:
        config.subplots.rows = int(np.ceil(n_subplots / config.subplots.cols))

    # handle init mode
    if config.init_mode == 'normal':
        if len(config.layout.updatemenus) > 0:
            config.layout.updatemenus[0]['active'] = 0
        config.default_trace.boxpoints = False
    elif config.init_mode == 'all':
        if len(config.layout.updatemenus) > 0:
            config.layout.updatemenus[0]['active'] = 1
        config.default_trace.boxpoints = 'all'
    elif config.init_mode == 'suspectedoutliers':
        if len(config.layout.updatemenus) > 0:
            config.layout.updatemenus[0]['active'] = 2
        config.default_trace.boxpoints = 'suspectedoutliers'
    elif config.init_mode == 'outliers':
        if len(config.layout.updatemenus) > 0:
            config.layout.updatemenus[0]['active'] = 3
        config.default_trace.boxpoints = 'outliers'

    # make figure with subplots
    fig = plotly.tools.make_subplots(**config.subplots)

    traces = []

    # interate over subplots
    for subplot_idx, subplot_data in enumerate(data):

        subplot_traces = []

        # create for each experiment a trace
        for data_idx, cur_data in enumerate(subplot_data):  # data source

            data_points = np.array([])
            elem_labels = []

            # collect data over elements
            for elem_idx, elem_data in enumerate(cur_data):  # data elements

                # get element data which could be in matrix format or array format
                if np.ndim(elem_data) == 1:
                    cur_elem_data = elem_data
                elif np.ndim(elem_data) == 2:
                    if elem_data.shape[0] == 1:
                        cur_elem_data = elem_data[0, :]
                    elif elem_data.shape[1] == 1:
                        cur_elem_data = elem_data[1, 0]
                    else:
                        raise ValueError('Invalid data format!')
                else:
                    raise ValueError('Invalid data format!')

                data_points = np.hstack((data_points, cur_elem_data))

                # handle trace for mean values
                group_label = config.default_group_label
                if len(config.group_labels) > elem_idx:
                    group_label = config.group_labels[elem_idx]
                group_label = ad.gui.jupyter.misc.replace_str_from_dict(str(group_label), {'group_idx': elem_idx})

                elem_labels.extend([group_label] * len(cur_elem_data))

            # handle trace for mean values
            trace_label = config.default_trace_label
            if len(config.trace_labels) > data_idx:
                trace_label = config.trace_labels[data_idx]
            trace_label = ad.gui.jupyter.misc.replace_str_from_dict(str(trace_label), {'data_idx': data_idx})

            trace_params = dict(
                x=elem_labels,
                y=data_points,
                name=trace_label,
            )

            trace_config = config.default_trace.copy()
            if len(config.default_subplot_traces) > subplot_idx:
                trace_config = ad.config.set_default_config(config.default_subplot_traces[subplot_idx], trace_config)
            if len(config.traces) > data_idx:
                trace_config = ad.config.set_default_config(config.traces[data_idx], trace_config)

            trace_params = ad.config.set_default_config(trace_config, trace_params)

            # handle legendgroup
            trace_legendgroup = trace_params.legendgroup
            if isinstance(trace_legendgroup, str):
                trace_legendgroup = ad.gui.jupyter.misc.replace_str_from_dict(trace_legendgroup, {'data_idx': data_idx,
                                                                                                  'subplot_idx': subplot_idx})
            trace_params.legendgroup = trace_legendgroup

            cur_trace = plotly.graph_objs.Box(**trace_params)
            subplot_traces.append(cur_trace)

        traces.append(subplot_traces)

    # set for the std toggle buttons which traces should be hidden and which ones should be shown
    layout = config.layout

    # set default values for all layouts
    def set_axis_properties_by_default(axis_name, fig_layout, config_layout):
        # sets the axis properties to default values

        def set_single_axis_property_default(cur_axis_name, default_name):
            if cur_axis_name in fig_layout or cur_axis_name in config_layout:
                cur_config = config_layout[cur_axis_name] if cur_axis_name in config_layout else dict()
                config_layout[cur_axis_name] = ad.config.set_default_config(cur_config, config_layout[default_name])

        default_name = 'default_' + axis_name

        set_single_axis_property_default(axis_name, default_name)
        set_single_axis_property_default(axis_name + '1', default_name)
        axis_idx = 2
        while True:
            cur_axis_name = axis_name + str(axis_idx)

            if cur_axis_name not in fig_layout and cur_axis_name not in config_layout:
                break

            set_single_axis_property_default(cur_axis_name, default_name)
            axis_idx += 1

    set_axis_properties_by_default('xaxis', fig['layout'], layout)
    set_axis_properties_by_default('yaxis', fig['layout'], layout)

    # remove default fields, because they are not true proerties of the plotly layout
    del (layout['default_xaxis'])
    del (layout['default_yaxis'])

    cur_row = 1
    cur_col = 1
    for subplot_idx in range(n_subplots):
        n_traces = len(traces[subplot_idx])

        fig.add_traces(traces[subplot_idx],
                       rows=[cur_row] * n_traces,
                       cols=[cur_col] * n_traces)

        if cur_col < config.subplots.cols:
            cur_col += 1
        else:
            cur_col = 1
            cur_row += 1

    fig['layout'].update(layout)

    plotly.offline.iplot(fig)

    return fig