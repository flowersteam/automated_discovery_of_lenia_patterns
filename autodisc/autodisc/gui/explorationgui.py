import autodisc as ad
from autodisc.gui.gui import BaseFrame
try:
    import tkinter as tk
except:
    import Tkinter as tk
from tkinter import ttk
import importlib
import warnings

class ExplorationGUI(BaseFrame):
    # TODO: ther seems to be a memory leak, altough a limit for the max_num_of_obs_in_memory is defined, the memory still inreases if more than that are observed

    @staticmethod
    def default_gui_config():
        default_gui_config = ad.gui.BaseFrame.default_gui_config()

        default_gui_config.dialog.title = 'Exploration Viewer'

        default_gui_config.is_get_obs_from_files = True
        default_gui_config.is_get_obs_by_experiment = True

        default_gui_config.experiment_num_of_steps = 100
        # default_gui_config.max_num_of_obs_in_memory = None

        default_gui_config.statistic_columns = []
        default_gui_config.detail_views = []

        return default_gui_config


    def __init__(self, master=None, explorer=None, datahandler=None, gui_config=None, **kwargs):
        '''

            Takes either an Explorer or a ExplorationDataHandler as input.
            Not both!

            gui_config:
                statistic_columns:
                    list
                        ['stat_idx']: Index of the statistic in the list of data.stats. Default = 0
                        ['stat_name']: Name of the statistic as under data.stats[stat_idx]['<stat_name>'].
                        ['disp_name']: Name of the stat displayed in the GUI. Default: <stat_name>

                detail_views:
                    list
                        ['type']: Either 'observations' or 'statistics'
                        ['gui']: name of gui that should be used, e.g. ExplorationExperimentGUI
                        ['gui_config']: Configuration of th gui. Default = []
                        ['disp_name']: Name used for the title of the window. Default:<gui>
        '''

        super().__init__(master=master, gui_config=gui_config, **kwargs)

        if explorer is not None and datahandler is not None:
            raise ValueError('Input can only be an Explorer or a DataHandler and not both!')

        if explorer is not None:
            self.explorer = explorer
            self.data = self.explorer.data

        if datahandler is not None:
            self.explorer = None
            self.data = datahandler

        num_of_detail_guis = len(self.gui_config.detail_views)
        self.exp_detail_guis = [None] * num_of_detail_guis

        # self.obs_ids_queue = None # holds ids of the results for which the observations are stored
        # self.reset_obs_in_memory()

        self.create_gui()
        self.display_data()

        if len(self.data) > 0:
            self.selected_run_id = list(self.data.runs.keys())[0]
            self.open_experiment_details(self.selected_run_id)
            self.tree.focus(self.selected_run_id)


    def create_gui(self):

        # make the treeview in the frame resizable
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(self,
                                 selectmode='browse')
        self.tree.grid(column=0, row=0, sticky=tk.NSEW)
        self.tree.bind('<<TreeviewSelect>>', self.on_treeview_select)

        self.scrollbar_y = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.scrollbar_y.grid(column=1, row=0, sticky=tk.NS)

        self.scrollbar_x = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.scrollbar_x.grid(column=0, row=1, sticky=tk.EW)

        self.tree.config(yscrollcommand=self.scrollbar_y.set)
        self.tree.config(xscrollcommand=self.scrollbar_x.set)


    def on_treeview_select(self, event):

        run_id = int(self.tree.focus())

        # show the experiment data in the property windows
        if run_id != self.selected_run_id:
            self.selected_run_id = run_id
            self.update_experiment_detail_guis(self.selected_run_id)


    def load_run_observations(self, run_id):

        obs = self.data[run_id].observations

        if obs is None and self.explorer is not None:

            [obs, statistics] = self.explorer.system.run(run_parameters=self.data[run_id].run_parameters,
                                                         stop_conditions=self.gui_config.experiment_num_of_steps)

            # save observations to immediate reuse it
            self.data.runs[run_id].observations = obs

        # # search if new obs is in memory queue
        # if self.obs_ids_queue:
        #
        #     if run_id in self.obs_ids_queue:
        #         # move the experiment_id to the top of the queue
        #         idx = self.obs_ids_queue.index(run_id)
        #         del(self.obs_ids_queue[idx])
        #     else:
        #         # remove last item and add new one
        #         if self.obs_ids_queue[-1]: # could be none
        #             del self.data[self.obs_ids_queue[-1]]['observations']
        #         self.obs_ids_queue.pop(-1)
        #
        #     self.obs_ids_queue.insert(0, run_id)


    # def reset_obs_in_memory(self):
    #     '''Sets the number of observations in the memory arcording to gui_config['max_num_of_obs_in_memory'].'''
    #
    #     if self.gui_config.max_num_of_obs_in_memory is None:
    #         # do nothing and store all obs
    #         self.obs_ids_queue = None
    #
    #     else:
    #
    #         self.obs_ids_queue = [None] * self.gui_config.max_num_of_obs_in_memory
    #
    #         if self.obs_ids_queue is None:
    #             # initialization --> have to check if the given experiment has too many observations, if so, remove them
    #
    #             idx = 0
    #
    #             for run_id, run_data in self.data.runs.items():
    #
    #                 if idx < self.gui_config.max_num_of_obs_in_memory and 'observations' in run_data and run_data.observations is not None:
    #                     self.obs_ids_queue[idx] = run_id
    #                     idx = idx + 1
    #                 else:
    #                     run_data['observations'] = None
    #
    #         else:
    #             if len(self.obs_ids_queue) > self.gui_config.max_num_of_obs_in_memory:
    #
    #                 # queue has to be shortened --> erase the connected observations
    #                 for idx in range(self.gui_config.max_num_of_obs_in_memory, len(self.obs_ids_queue)):
    #                     self.data.runs[self.obs_ids_queue[idx]]['observations'] = None
    #
    #                 del(self.obs_ids_queue[(self.gui_config.max_num_of_obs_in_memory):len(self.obs_ids_queue)])
    #
    #             elif len(self.obs_ids_queue) < self.gui_config.max_num_of_obs_in_memory:
    #
    #                 # queue has to be extended
    #                 self.obs_ids_queue.extend([None] * self.gui_config.max_num_of_obs_in_memory-len(self.obs_ids_queue))


    def open_experiment_details(self, run_id):

        # load observations if they dont exist
        self.load_run_observations(run_id)

        for exp_detail_idx in range(len(self.gui_config.detail_views)):
            if not self.exp_detail_guis[exp_detail_idx]:
                self.open_experiment_detail_window(exp_detail_idx, run_id)

        self.update_experiment_detail_guis(run_id)


    def open_experiment_detail_window(self, detail_gui_idx, run_id):
        detail_config = self.gui_config.detail_views[detail_gui_idx]

        if self.exp_detail_guis[detail_gui_idx]:
            self.exp_detail_guis[detail_gui_idx].master.destroy()

        # load class as defined in config
        module_name = '.'.join(detail_config['gui'].split('.')[0:-1])
        module = importlib.import_module(module_name)

        class_name = detail_config['gui'].split('.')[-1]
        gui_class = getattr(module, class_name)

        gui_config = detail_config.get('gui_config', [])

        self.exp_detail_guis[detail_gui_idx] = gui_class(master=self,
                                                         is_dialog=True,
                                                         gui_config=gui_config)

        self.exp_detail_guis[detail_gui_idx].display_exploration_data(self.data, run_id)


    def update_experiment_detail_guis(self, run_id):

        self.load_run_observations(run_id)

        for exp_detail_gui_idx in range(len(self.gui_config.detail_views)):
            if self.exp_detail_guis[exp_detail_gui_idx]:
                self.exp_detail_guis[exp_detail_gui_idx].display_exploration_data(self.data, run_id)


    def display_data(self):

        # remove existing data in treeview
        self.tree.delete(*self.tree.get_children())

        if len(self.data) > 0:

            # add columns for the statistics that should be displayed
            stat_names = []
            col_names = []
            format_strs = []

            for statistic_column in self.gui_config.statistic_columns:

                stat_name = statistic_column['stat_name']
                stat_names.append(stat_name)

                name = statistic_column.get('disp_name', stat_name)
                col_names.append(name)

                format_str = statistic_column.get('format', None)
                format_strs.append(format_str)

            if col_names:
                self.tree['columns'] = tuple(range(len(col_names)))

                for col_idx in range(len(col_names)):
                    self.tree.heading(col_idx, text=col_names[col_idx])

            # go through data
            for run_data in self.data:

                # add columns for the statistics that should be displayed
                stat_values = []
                for col_idx in range(len(stat_names)):

                    val = run_data.statistics[stat_names[col_idx]]

                    if format_strs[col_idx]:
                        val_str = format_strs[col_idx].format(val)
                    else:
                        val_str = str(val)

                    stat_values.append(val_str)

                # show the name of the parameter set if it has one, otherwise its id
                if 'name' in run_data:
                    text = '({}) {}'.format(run_data.id, run_data.name)
                else:
                    text = str(run_data.id)

                self.tree.insert('', 'end', run_data.id, text=text, values=stat_values)


    def run(self):
        self.master.mainloop()