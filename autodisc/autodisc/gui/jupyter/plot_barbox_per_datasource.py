import autodisc

def plot_barbox_per_datasource(data=None, experiment_ids=None, data_source=None, repetition_ids=None, config=None, **kwargs):
    default_config = dict(

        plot_type='plotly_meanstd_bar',  # 'plotly_meanstd_bar', 'plotly_box'

        default_trace_label='<datasource>',
        trace_labels=[],

        default_group_label='<exp_id>',
        group_labels=[],
    )
    config = autodisc.config.set_default_config(kwargs, config, default_config)

    # load data
    plot_data, experiment_ids = autodisc.gui.jupyter.misc.get_datasource_data(data=data,
                                                                              experiment_ids=experiment_ids,
                                                                              data_source=data_source,
                                                                              repetition_ids=repetition_ids)

    if data_source is None:
        data_source = ['']

    # create the labels
    new_trace_labels = []
    for datasource_idx, cur_data_source in enumerate(data_source):
        trace_label = config.default_trace_label
        if len(config.trace_labels) > datasource_idx:
            trace_label = config.trace_labels[datasource_idx]
        trace_label = autodisc.gui.jupyter.misc.replace_str_from_dict(str(trace_label), {'datasource_idx': datasource_idx,
                                                                                         'datasource': str(cur_data_source)})
        new_trace_labels.append(trace_label)
    config.trace_labels = new_trace_labels

    new_group_labels = []
    for experiment_idx, experiment_id in enumerate(experiment_ids):
        group_label = config.default_group_label
        if len(config.group_labels) > experiment_idx:
            group_label = config.group_labels[experiment_idx]
        group_label = autodisc.gui.jupyter.misc.replace_str_from_dict(str(group_label), {'exp_idx': experiment_idx,
                                                                                         'exp_id': experiment_id})
        new_group_labels.append(group_label)
    config.group_labels = new_group_labels

    if config.plot_type == 'plotly_meanstd_bar':
        return autodisc.gui.jupyter.plotly_meanstd_bar(data=plot_data, config=config)
    elif config.plot_type == 'plotly_box':
        return autodisc.gui.jupyter.plotly_box(data=plot_data, config=config)