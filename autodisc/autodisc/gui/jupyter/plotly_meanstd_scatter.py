import autodisc as ad
import numpy as np
import plotly

def plotly_meanstd_scatter(data=None, config=None, **kwargs):
    '''
    param repetition_ids: Either scalar int with single id, list with several that are used for each experiment, or a dict with repetition ids per experiment.
    '''
    default_config = dict(
        subplots=dict(
            rows=None,
            cols=None,
            print_grid=False
        ),
        std=dict(
            style='shaded',  # or errorbar
            steps=1,
            visible=True
        ),
        init_mode='mean_std',  # mean_std, mean, elements
        plotly_format = 'webgl',   # webgl or svg
        layout=dict(

            default_xaxis=dict(),  # if several subplots, then these are the default values for all xaxis config in fig.layout
            default_yaxis=dict(),  # if several subplots, then these are the default values for all yaxis config in fig.layout

            xaxis=dict(),
            yaxis=dict(),

            updatemenus=[
                dict(type="buttons",
                     active=0,
                     buttons=[
                         dict(label='mean + std',
                              method='restyle',
                              args=[{'visible': []}]),
                         dict(label='mean',
                              method='restyle',
                              args=[{'visible': []}]),
                         dict(label='elements',
                              method='restyle',
                              args=[{'visible': []}]),
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

        default_trace=dict(),

        default_mean_trace=dict(
            legendgroup='<subplot_idx>-<data_idx>',  # subplot_idx, data_idx
            hoverinfo='text+x',
        ),
        default_subplot_mean_traces=[],  # default config of traces per subplot
        mean_traces=[],

        default_std_trace=dict(
            legendgroup='<mean_trace_legendgroup>',  # subplot_idx, data_idx, mean_trace_legendgroup
            hoverinfo='none',
            showlegend=False,
        ),
        default_subplot_std_traces=[],  # default config of traces per subplot
        std_traces=[],

        default_element_trace=dict(  # overall default
            legendgroup=None,  # subplot_idx, data_idx, elem_idx, subelem_idx, mean_trace_legendgroup, std_trace_legendgroup
        ),
        default_subplot_element_traces=[],  # default per subplot
        default_data_element_traces=[],  # default per data item
        element_traces=[],  # individual items

        default_mean_label='<data_idx>',
        mean_labels=[],

        default_element_label='<mean_label> - <subelem_idx>',  # possible replacements: <mean_label>, <subelem_idx>, <elem_idx>, <data_idx>
        element_labels=[],

        default_colors=plotly.colors.DEFAULT_PLOTLY_COLORS,
    )
    config = ad.config.set_default_config(kwargs, config, default_config)

    if data is None:
        data = np.array([])

    # format data in form [subplot_idx:list][trace_idx:list][elems_per_trace:numpy.ndarray]
    if isinstance(data, np.ndarray):
        data = [[data]]
    elif isinstance(data[0], np.ndarray):
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

    if config.plotly_format.lower() == 'webgl':
        plotly_scatter_plotter = plotly.graph_objs.Scattergl
    elif config.plotly_format.lower() == 'svg':
        plotly_scatter_plotter = plotly.graph_objs.Scatter
    else:
        raise ValueError('Unknown config {!r} for plotly_format! Allowed values: \'webgl\', \'svg\'.')

    # make figure with subplots
    fig = plotly.tools.make_subplots(**config.subplots)

    mean_traces = []
    elem_traces = []

    elem_idx = 0

    # interate over subplots
    for subplot_idx, subplot_data in enumerate(data):

        subplot_mean_traces = []
        subplot_elem_traces = []

        # create for each experiment a trace
        for data_idx, cur_data in enumerate(subplot_data):

            mean_data = np.nanmean(cur_data, axis=0)
            std_data = np.nanstd(cur_data, axis=0)

            # TODO: allow setting of custom x values
            # this can not simply be done by seeting the x attribute of the trace, because the std trace has an extra
            x_values = list(range(len(mean_data)))

            # handle trace for mean values

            info_text = ['{} ± {}'.format(mean_data[idx], std_data[idx]) for idx in range(len(mean_data))]

            mean_label = config.default_mean_label
            if len(config.mean_labels) > data_idx:
                mean_label = config.mean_labels[data_idx]
            mean_label = ad.gui.jupyter.misc.replace_str_from_dict(str(mean_label), {'data_idx': data_idx})

            mean_trace_params = dict(
                x=x_values,
                y=mean_data,
                line=dict(color=config.default_colors[data_idx % len(config.default_colors)]),
                name=mean_label,
                text=info_text,
            )

            mean_trace_config = ad.config.set_default_config(config.default_mean_trace, config.default_trace)
            if len(config.default_subplot_mean_traces) > subplot_idx:
                mean_trace_config = ad.config.set_default_config(config.default_subplot_mean_traces[subplot_idx], mean_trace_config)
            if len(config.mean_traces) > data_idx:
                mean_trace_config = ad.config.set_default_config(config.mean_traces[data_idx], mean_trace_config)

            mean_trace_params = ad.config.set_default_config(mean_trace_config, mean_trace_params)

            # handle legendgroup
            mean_trace_legendgroup = mean_trace_params.legendgroup
            if isinstance(mean_trace_legendgroup, str):
                mean_trace_legendgroup = ad.gui.jupyter.misc.replace_str_from_dict(mean_trace_legendgroup, {'data_idx': data_idx,
                                                                                                            'subplot_idx': subplot_idx})
            mean_trace_params.legendgroup = mean_trace_legendgroup

            cur_mean_trace = plotly_scatter_plotter(**mean_trace_params)
            subplot_mean_traces.append(cur_mean_trace)

            # handle trace for std values

            if config.std.style.lower() == 'shaded':

                fill_color = config.default_colors[data_idx % len(config.default_colors)]
                fill_color = fill_color.replace('rgb', 'rgba')
                fill_color = fill_color.replace(')', ', 0.2)')

                std_trace_params = dict(
                    x=x_values + x_values[::-1],
                    y=np.hstack((mean_data + std_data, mean_data[::-1] - std_data[::-1])),
                    fill='tozerox',
                    line=dict(color='rgba(255,255,255,0)'),
                    fillcolor=fill_color,
                )

            elif config.std.style.lower() == 'errorbar':

                std_trace_params = dict(
                    x=x_values[::config.std.steps],
                    y=mean_data[::config.std.steps],
                    error_y=dict(type='data', array=std_data, visible=True),
                    mode='markers',
                    line=dict(color=config.default_colors[data_idx % len(config.default_colors)]),
                    marker=dict(size=0, opacity=0),
                )

            else:
                raise ValueError('Unknown config.std.style ({!r})! Options: \'shaded\', \'errorbar\''.format(config.std.type))

            std_trace_config = ad.config.set_default_config(config.default_std_trace, config.default_trace)
            if len(config.default_subplot_std_traces) > subplot_idx:
                std_trace_config = ad.config.set_default_config(config.default_subplot_std_traces[subplot_idx], std_trace_config)
            if len(config.std_traces) > data_idx:
                std_trace_config = ad.config.set_default_config(config.std_traces[data_idx], std_trace_config)
            std_trace_params = ad.config.set_default_config(std_trace_config, std_trace_params)

            # handle legendgroup
            std_trace_legendgroup = std_trace_params.legendgroup
            if isinstance(std_trace_legendgroup, str):
                std_trace_legendgroup = ad.gui.jupyter.misc.replace_str_from_dict(std_trace_legendgroup, {'data_idx': data_idx,
                                                                                                          'subplot_idx': subplot_idx,
                                                                                                          'mean_trace_legendgroup': mean_trace_legendgroup}
                                                              )
            std_trace_params.legendgroup = std_trace_legendgroup

            cur_std_trace = plotly_scatter_plotter(**std_trace_params)
            subplot_mean_traces.append(cur_std_trace)

            # traces for each data element
            n_elems = cur_data.shape[0]
            color_coeff_step = 1 / n_elems
            cur_color_coeff = 0 + color_coeff_step
            for cur_elem_idx in range(n_elems):

                cur_elem_data = cur_data[cur_elem_idx, :]

                element_label = config.default_element_label
                if len(config.element_labels) > data_idx:
                    element_label = config.element_labels[data_idx]
                element_label = ad.gui.jupyter.misc.replace_str_from_dict(str(element_label), {'data_idx': data_idx,
                                                                                               'subelem_idx': cur_elem_idx,
                                                                                               'elem_idx': elem_idx,
                                                                                               'mean_label': mean_label})

                color = ad.gui.jupyter.misc.transform_color_str_to_tuple(config.default_colors[data_idx % len(config.default_colors)])
                color = (color[0],
                         int(color[1] * cur_color_coeff),
                         int(color[2] * cur_color_coeff),
                         int(color[3] * cur_color_coeff))
                color = ad.gui.jupyter.misc.transform_color_tuple_to_str(color)
                cur_color_coeff += color_coeff_step

                element_trace_params = dict(
                    x=x_values,
                    y=cur_data[cur_elem_idx, :],
                    line=dict(color=color),
                    name=element_label,
                    visible=True,
                )

                element_trace_config = ad.config.set_default_config(config.default_element_trace, config.default_trace)
                if len(config.default_subplot_element_traces) > subplot_idx:
                    element_trace_config = ad.config.set_default_config(config.default_subplot_element_traces[subplot_idx], element_trace_config)
                if len(config.default_data_element_traces) > cur_elem_idx:
                    element_trace_config = ad.config.set_default_config(config.default_data_element_traces[cur_elem_idx], element_trace_config)
                if len(config.element_traces) > elem_idx:
                    element_trace_config = ad.config.set_default_config(config.element_traces[elem_idx], element_trace_config)

                element_trace_params = ad.config.set_default_config(element_trace_config, element_trace_params)

                # handle legendgroup
                element_trace_legendgroup = element_trace_params.legendgroup
                if isinstance(element_trace_legendgroup, str):
                    element_trace_legendgroup = ad.gui.jupyter.misc.replace_str_from_dict(element_trace_legendgroup, {'subelem_idx': cur_elem_idx,
                                                                                                                      'elem_idx': elem_idx,
                                                                                                                      'data_idx': data_idx,
                                                                                                                      'subplot_idx': subplot_idx,
                                                                                                                      'mean_trace_legendgroup': mean_trace_legendgroup,
                                                                                                                      'std_trace_legendgroup': std_trace_legendgroup}
                                                                      )
                element_trace_params.legendgroup = element_trace_legendgroup

                cur_elem_trace = plotly_scatter_plotter(**element_trace_params)
                subplot_elem_traces.append(cur_elem_trace)

                elem_idx += 1

        mean_traces.append(subplot_mean_traces)
        elem_traces.append(subplot_elem_traces)

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

    update_menus_visible_meanstd = []
    update_menus_visible_mean = []
    update_menus_visible_elements = []

    for subplot_idx in range(len(mean_traces)):
        update_menus_visible_meanstd.extend([True, True] * int(len(mean_traces[subplot_idx]) / 2) + [False] * int(len(elem_traces[subplot_idx])))
        update_menus_visible_mean.extend([True, False] * int(len(mean_traces[subplot_idx]) / 2) + [False] * int(len(elem_traces[subplot_idx])))

        element_default_visibility = [elem_trace['visible'] for elem_trace in elem_traces[subplot_idx]]
        update_menus_visible_elements.extend([False, False] * int(len(mean_traces[subplot_idx]) / 2) + element_default_visibility)

    if layout.updatemenus:

        layout.updatemenus[0]['buttons'][0]['args'][0]['visible'] = update_menus_visible_meanstd
        layout.updatemenus[0]['buttons'][1]['args'][0]['visible'] = update_menus_visible_mean
        layout.updatemenus[0]['buttons'][2]['args'][0]['visible'] = update_menus_visible_elements

        if config.init_mode == 'mean_std':
            config.layout.updatemenus[0]['active'] = 0
        elif config.init_mode == 'mean':
            config.layout.updatemenus[0]['active'] = 1
        elif config.init_mode == 'elements':
            config.layout.updatemenus[0]['active'] = 2
        else:
            raise ValueError('Value {!r} for \'config.init_mode\' is not supported! Only \'mean_std\',\'mean\',\'elements\'.'.format(config.init_mode))


    if config.init_mode == 'mean_std':
        trace_visibility = update_menus_visible_meanstd
    elif config.init_mode == 'mean':
        trace_visibility = update_menus_visible_mean
    elif config.init_mode == 'elements':
        trace_visibility = update_menus_visible_elements
    else:
        raise ValueError('Value {!r} for \'config.init_mode\' is not supported! Only \'mean_std\',\'mean\',\'elements\'.'.format(config.init_mode))


    cur_row = 1
    cur_col = 1
    for subplot_idx in range(n_subplots):

        n_traces = len(mean_traces[subplot_idx]) + len(elem_traces[subplot_idx])

        fig.add_traces(mean_traces[subplot_idx] + elem_traces[subplot_idx],
                       rows=[cur_row] * n_traces,
                       cols=[cur_col] * n_traces)

        if cur_col < config.subplots.cols:
            cur_col += 1
        else:
            cur_col = 1
            cur_row += 1

    for trace_idx in range(len(fig['data'])):
        fig['data'][trace_idx]['visible'] = trace_visibility[trace_idx]

    fig['layout'].update(layout)

    plotly.offline.iplot(fig)

    return fig