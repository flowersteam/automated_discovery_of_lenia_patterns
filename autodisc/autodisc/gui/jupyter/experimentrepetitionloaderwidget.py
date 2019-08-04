import autodisc as ad
import ipywidgets

class ExperimentRepetitionLoaderWidget(ipywidgets.Box):

    @staticmethod
    def get_default_gui_config():
        default_config = ad.Config()

        default_config.box_layout = ad.Config()
        default_config.box_layout.display = 'stretch'
        default_config.box_layout.align_items = 'stretch'

        default_config.experiment_selection = ad.Config()
        default_config.experiment_selection.description = 'Experiment:'
        default_config.experiment_selection.layout = ad.Config()
        default_config.experiment_selection.layout.flex = '1 1 0%'
        default_config.experiment_selection.layout.width = 'auto'

        default_config.repetition_selection = ad.Config()
        default_config.repetition_selection.description = 'Repetition:'
        default_config.repetition_selection.layout = ad.Config()
        default_config.repetition_selection.layout.flex = '1 1 0%'
        default_config.repetition_selection.layout.width = 'auto'

        default_config.load_button = ad.Config()
        default_config.load_button.layout = ad.Config()
        default_config.load_button.layout.width = '150px'
        default_config.load_button.description = 'Load'
        default_config.load_button.disabled = False
        default_config.load_button.button_style = ''  # 'success', 'info', 'warning', 'danger' or ''
        default_config.load_button.tooltip = 'Load the data for the selected experiment and repetition.'

        return default_config

    def __init__(self, experiment_definitions, repetition_ids, load_function, config=None, **kwargs):
        self.config = ad.config.set_default_config(kwargs, config, ExperimentRepetitionLoaderWidget.get_default_gui_config())

        self.data = []
        self.experiment_definitions = experiment_definitions
        self.repetition_ids = repetition_ids
        self.load_function = load_function

        self.experiment_ids = [exp_def['id'] for exp_def in self.experiment_definitions]

        # create Experiment selection
        self.exp_selection_widget = ipywidgets.Dropdown(
            options=self.experiment_ids,
            value=self.experiment_ids[0],
            **self.config.experiment_selection
        )

        # create Repetition selection
        self.rep_selection_widget = ipywidgets.Dropdown(
            options=list(repetition_ids),
            value=list(repetition_ids)[0],
            **self.config.repetition_selection
        )

        # create Load button
        self.load_btn = ipywidgets.Button(
            **self.config.load_button
        )

        self.load_btn.on_click(self.load_data)

        return super().__init__([self.exp_selection_widget, self.rep_selection_widget, self.load_btn], layout=self.config.box_layout)

    def load_data(self, btn):
        self.data = self.load_function(
            self.exp_selection_widget.value,
            self.rep_selection_widget.value
        )
