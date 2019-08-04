import autodisc as ad
from autodisc.gui.gui import BaseFrame
try:
    import tkinter as tk
except:
    import Tkinter as tk


class ExplorationParametersGUI(BaseFrame):

    @staticmethod
    def default_gui_config():

        def_config = ad.gui.BaseFrame.default_gui_config()

        def_config.dialog.title = 'Exploration Parameters'
        
        def_config.show_system_parameters = True
        def_config.show_run_parameters = True
        def_config.system_parameters_config = [{'type': 'table', 'parameters': 'all'}]
        def_config.run_parameters_config = [{'type': 'table', 'parameters': 'all'}]

        def_config.system_parameter_label = ad.Config({'text': 'System Parameters:'})
        def_config.run_parameter_label = ad.Config({'text': 'Run Parameters:'})
        return def_config


    def __init__(self, master=None, gui_config=None, **kwargs):
        super().__init__(master=master, gui_config=gui_config, **kwargs)

        self.system_parameters = None
        self.run_parameters = None

        self.create_gui()


    def create_gui(self):

        # make the elements in x direction resizable
        self.columnconfigure(0, weight=1)

        self.gui_elements = []
        self.system_parameter_update_funcs = []
        self.run_parameter_update_funcs = []

        row_idx = 0

        # create elements for the system parameters
        if self.gui_config.show_system_parameters:

            # create a label
            self.param_sys_label = tk.Label(master=self, **self.gui_config.system_parameter_label)
            self.param_sys_label.grid(row=row_idx, column=0, sticky=tk.NW)
            row_idx = row_idx + 1

            self.system_parameter_update_funcs, row_idx = self.create_category_windows_elements(config=self.gui_config.system_parameters_config,
                                                                                                row_idx=row_idx)

        # create elements for the run parameters
        if self.gui_config.show_run_parameters:

            # create a label
            self.param_run_label = tk.Label(master=self, **self.gui_config.run_parameter_label)
            self.param_run_label.grid(row=row_idx, column=0, sticky=tk.NW)
            row_idx = row_idx + 1

            self.run_parameter_update_funcs, row_idx = self.create_category_windows_elements(config=self.gui_config.run_parameters_config,
                                                                                             row_idx=row_idx)


    def create_category_windows_elements(self, config, row_idx):

        update_funcs = []

        for param_config in config:

            if param_config['type'] is 'table':
                sub_frame = ad.gui.DictTableGUI(master=self,
                                                gui_config=param_config.get('gui_config', None))
                update_funcs.append(sub_frame.display_data)

            if param_config['type'] is 'image':
                sub_frame = ad.gui.ArrayToImageGUI(master=self,
                                                   gui_config=param_config.get('gui_config', None))

                func = lambda params: self.update_images_wrapper(sub_frame, params)
                update_funcs.append(func)
                #update_funcs.append(image_gui.update_data)

            sub_frame.grid(row=row_idx, column=0, sticky=tk.NSEW)
            self.rowconfigure(row_idx, weight=1) # allow the sub_frame to resize if windows in made smaller
            row_idx = row_idx + 1

            sub_frame.columnconfigure(0, weight=1)
            sub_frame.rowconfigure(0, weight=1)

            self.gui_elements.append(sub_frame)

        return update_funcs, row_idx


    def display_exploration_data(self, exploration_data, run_id):
        system_parameters = exploration_data.system_parameters
        run_parameters = exploration_data.runs[run_id].run_parameters
        self.display_data(system_parameters=system_parameters, run_parameters=run_parameters)


    def display_data(self, system_parameters=None, run_parameters=None):

        if system_parameters is not None:
            self.system_parameters = system_parameters.copy()

        if run_parameters is not None:
            self.run_parameters = run_parameters.copy()

        # remove all system_parameters that are again specified in the run parameters
        if self.system_parameters is not None and self.run_parameters is not None:
            for key in self.run_parameters.keys():
                if key in self.system_parameters:
                    del self.system_parameters[key]


        if self.gui_config.show_system_parameters:
            self.update_category(config=self.gui_config.system_parameters_config,
                                 update_funcs=self.system_parameter_update_funcs,
                                 params=self.system_parameters)


        if self.gui_config.show_run_parameters:
            self.update_category(config=self.gui_config.run_parameters_config,
                                 update_funcs=self.run_parameter_update_funcs,
                                 params=self.run_parameters)


    def update_category(self, config, update_funcs, params):

        for params_gui_idx in range(len(config)):

            params_gui_config = config[params_gui_idx]

            cur_params = self.filter_parameters(params, params_gui_config)

            func = update_funcs[params_gui_idx]
            func(cur_params)


    def filter_parameters(self, params, config):

        filtered_params = {}

        # get the data that should be displayed
        if config['parameters'] is 'all':
            filtered_params = params
        else:
            for param_config in config['parameters']:
                param_name = param_config['name']
                filtered_params[param_name] = params[param_name]

        return filtered_params


    def update_images_wrapper(self, gui, params):

        # converts the params to the input parameters of the ImageArrayGUI which are lists of images and names
        image_data = list(params.values())
        image_names = [name + ':' for name in params.keys()]
        gui.display_data(image_data, image_names)