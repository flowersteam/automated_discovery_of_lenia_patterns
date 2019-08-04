# SELECTION WIDGET
import collections
import numpy as np
import ipywidgets

def interact_selection_multiple_experiments_repetitions(func, experiment_definitions=None, experiment_ids=None, repetition_ids=None, **kwargs):
    if experiment_definitions is not None and experiment_ids is not None:
        raise ValueError('Only experiment_definitions or experiment_ids are allowed as parameters!')

    # lists with information for each experiment

    if experiment_ids is not None:
        # experiment_names = [[]] * len(experiment_ids)
        experiment_is_default = [True] * len(experiment_ids)
    else:
        # experiment_names = [] # names
        experiment_is_default = []  # default values

        if experiment_definitions is not None:

            experiment_ids = []

            for exp_def in experiment_definitions:
                experiment_ids.append(exp_def['id'])

                #                 if 'name' in exp_def:
                #                     experiment_names.append(exp_def['name'])
                #                 else:
                #                     experiment_names.append([])

                if 'is_default' in exp_def:
                    experiment_is_default.append(exp_def['is_default'])
                else:
                    experiment_is_default.append(True)

    if repetition_ids is None:
        repetition_ids = []

    func_paramaters = kwargs

    exp_ids = collections.OrderedDict()
    exp_options_dict = collections.OrderedDict()

    is_all_default = np.all(experiment_is_default)

    exp_options_dict['exp_all'] = ipywidgets.Checkbox(description='all', value=bool(~is_all_default))
    exp_ids['exp_all'] = 'all'

    for experiment_idx, experiment_id in enumerate(experiment_ids):
        idx_name = 'exp_' + str(experiment_id)

        #         if experiment_names[experiment_idx]:
        #             descr_str = '{} - {}'.format(experiment_id, experiment_names[experiment_idx])
        #         else:
        #             descr_str = '{}'.format(experiment_id)
        descr_str = '{}'.format(experiment_id)

        if is_all_default:
            is_on = False
        else:
            is_on = experiment_is_default[experiment_idx]

        exp_options_dict[idx_name] = ipywidgets.Checkbox(description=descr_str, value=is_on)
        exp_ids[idx_name] = experiment_id

    exp_multi_checkbox_widget = ipywidgets.VBox(list(exp_options_dict.values()), layout={'overflow': 'scroll'})

    rep_ids = collections.OrderedDict()
    rep_options_dict = collections.OrderedDict()

    rep_options_dict['rep_all'] = ipywidgets.Checkbox(description='all', value=True)
    rep_ids['rep_all'] = 'all'

    for repetition_id in repetition_ids:
        idx_name = 'rep_' + str(repetition_id)
        rep_options_dict[idx_name] = ipywidgets.Checkbox(description=str(repetition_id), value=False)
        rep_ids[idx_name] = repetition_id

    rep_multi_checkbox_widget = ipywidgets.VBox(list(rep_options_dict.values()), layout={'overflow': 'scroll'})

    accordion_widget = ipywidgets.Accordion(children=[exp_multi_checkbox_widget, rep_multi_checkbox_widget],
                                            selected_index=None)
    accordion_widget.set_title(0, 'Experiments')
    accordion_widget.set_title(1, 'Repetitions')

    def internal_func(**kwargs):

        cur_exp_ids = []
        for arg_name, arg_value in exp_options_dict.items():
            if arg_value.value:
                exp_id = exp_ids[arg_name]
                if exp_id == 'all':
                    cur_exp_ids = [exp_id]
                    break
                else:
                    cur_exp_ids.append(exp_id)

        cur_rep_ids = []
        for arg_name, arg_value in rep_options_dict.items():
            if arg_value.value:
                rep_id = rep_ids[arg_name]
                if rep_id == 'all':
                    cur_rep_ids = [rep_id]
                    break
                else:
                    cur_rep_ids.append(rep_id)

        func(experiment_ids=cur_exp_ids, repetition_ids=cur_rep_ids, **func_paramaters)

    out = ipywidgets.interactive_output(internal_func, {**exp_options_dict, **rep_options_dict})

    display(out, accordion_widget)

    # hack: I need to change the value of one control element to force the plot to replot
    # otherwise it will not be shown in the beginning
    exp_options_dict['exp_all'].value = bool(is_all_default)

    return out, accordion_widget