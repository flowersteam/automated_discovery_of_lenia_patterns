import autodisc as ad
from autodisc.gui.gui import DictTableGUI, BaseFrame
import collections
try:
    import tkinter as tk
except:
    import Tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class StatisticTableGUI(DictTableGUI):

    @staticmethod
    def default_gui_config():

        def_config = DictTableGUI.default_gui_config()

        def_config.dialog.title = 'Statistics'
        def_config.statistics = ad.Config()

        return def_config


    def display_exploration_data(self, exploration_data, run_id):
        self.display_data(exploration_data.runs[run_id].statistics)


    def display_data(self, statistics):

        if not isinstance(statistics, collections.abc.Sequence):
            self.statistics = [statistics]
        else:
            self.statistics = statistics

        # get the data from the stats that should be displayed
        data = collections.OrderedDict()

        for stat_config in self.gui_config.statistics:

            stat_name = stat_config['name']
            disp_name = stat_config.get('disp_name', stat_name)
            stats_idx = stat_config.get('stats_idx', 0)
            format_str = stat_config.get('format', None)

            stat_data = self.statistics[stats_idx][stat_name]
            if format_str:
                stat_data = format_str.format(stat_data)

            data[disp_name] = stat_data

        super().display_data(data)



class StatisticLineGUI(BaseFrame):

    @staticmethod
    def default_gui_config():
        default_config = BaseFrame.default_gui_config()
        default_config.figure = {'figsize':(6,5), 'dpi':100}
        default_config.legend = {'loc': 'upper right'}
        default_config.statistics = ad.Config()
        return default_config


    def __init__(self, master=None, gui_config=None, **kwargs):
        super().__init__(master=master, gui_config=gui_config, **kwargs)

        self.create_gui()


    def create_gui(self):

        # make the treeview in the frame resizable
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.figure = plt.Figure(**self.gui_config['figure'])
        self.ax = self.figure.add_subplot(111)
        self.legend = None

        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.figure_canvas.get_tk_widget().grid(sticky=tk.NSEW)

        self.master.update_idletasks()


    def display_exploration_data(self, exploration_data, run_id):
        self.display_data(exploration_data.runs[run_id].statistics)


    def display_data(self, statistics):

        if not isinstance(statistics, collections.abc.Sequence):
            self.statistics = [statistics]
        else:
            self.statistics = statistics

        self.ax.clear()

        for stat_config in self.gui_config.statistics:

            stat_name = stat_config['name']
            disp_name = stat_config.get('disp_name', stat_name)
            stats_idx = stat_config.get('stats_idx', 0)

            stat_data = self.statistics[stats_idx][stat_name]

            self.ax.plot(stat_data, label=disp_name)

        self.legend = self.ax.legend(**self.gui_config.legend)

        self.figure_canvas.draw_idle()



class StatisticBarGUI(BaseFrame):

    @staticmethod
    def default_gui_config():
        default_config = BaseFrame.default_gui_config()
        default_config.figure = {'figsize':(6,5), 'dpi':100}
        default_config.legend = {'loc': 'upper right'}
        default_config.statistics = ad.Config()
        return default_config


    def __init__(self, master=None, gui_config=None, **kwargs):
        super().__init__(master=master, gui_config=gui_config, **kwargs)

        self.create_gui()


    def create_gui(self):

        # make the treeview in the frame resizable
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.figure = plt.Figure(**self.gui_config['figure'])
        self.ax = self.figure.add_subplot(111)
        self.legend = None

        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.figure_canvas.get_tk_widget().grid(sticky=tk.NSEW)

        self.master.update_idletasks()


    def display_exploration_data(self, exploration_data, run_id):
        self.display_data(exploration_data.runs[run_id].statistics)


    def display_data(self, statistics):

        if not isinstance(statistics, collections.abc.Sequence):
            self.statistics = [statistics]
        else:
            self.statistics = statistics

        self.ax.clear()

        stat_names = []
        for stat_config_idx in range(len(self.gui_config.statistics)):
            stat_config = self.gui_config.statistics[stat_config_idx]
            stat_name = stat_config['name']
            disp_name = stat_config.get('disp_name', stat_name)
            stats_idx = stat_config.get('stats_idx', 0)

            stat_data = self.statistics[stats_idx][stat_name]

            stat_names.append(disp_name)
            self.ax.bar(stat_config_idx, stat_data, align='center')

        self.ax.set_xticks(range(len(stat_names)))
        self.ax.set_xticklabels(stat_names)

        self.figure_canvas.draw_idle()