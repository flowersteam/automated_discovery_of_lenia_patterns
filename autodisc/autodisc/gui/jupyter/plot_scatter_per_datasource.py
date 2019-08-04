import autodisc

def plot_scatter_per_datasource(experiment_ids=None, repetition_ids=None, data_source=None, data=None, config=None, **kwargs):
    default_config = dict(
        default_mean_label='<exp_id>',
        mean_labels=[]
    )
    config = autodisc.config.set_default_config(kwargs, config, default_config)

    # load data
    if experiment_ids is None:
        experiment_ids = ['all']
    elif not isinstance(experiment_ids, list):
        experiment_ids = [experiment_ids]
    if experiment_ids == ['all']:
        experiment_ids = list(data.keys())

    # handle data source --> make to list if not list
    if data_source is not None and not isinstance(data_source, list):
        data_source = [data_source]

    # create for each experiment a trace
    plot_data = []

    for cur_data_source in data_source:

        cur_data_source_data = []

        for experiment_id in experiment_ids:
            cur_data_source_data.append(autodisc.gui.jupyter.misc.get_experiment_data(data=data, experiment_id=experiment_id, data_source=cur_data_source, repetition_ids=repetition_ids))

        plot_data.append(cur_data_source_data)

    # create the labels
    new_mean_labels = []
    for data_idx, experiment_id in enumerate(experiment_ids):

        mean_label = config.default_mean_label
        if len(config.mean_labels) > data_idx:
            mean_label = config.mean_labels[data_idx]
        mean_label = autodisc.gui.jupyter.misc.replace_str_from_dict(str(mean_label), {'exp_idx': data_idx,
                                                                                       'exp_id': experiment_id})

        new_mean_labels.append(mean_label)
    config.mean_labels = new_mean_labels

    return autodisc.gui.jupyter.plotly_meanstd_scatter(data=plot_data, config=config)
