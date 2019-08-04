import autodisc as ad
import numpy as np
from autodisc.gui import BaseFrame, AutoScrollbar
try:
    import tkinter as tk
except:
    import Tkinter as tk
from tkinter import ttk
import math

class DataViewerGUI(BaseFrame):

    @staticmethod
    def default_gui_config():
        default_gui_config = BaseFrame.default_gui_config()

        default_gui_config['dialog']['title'] = 'DataViewerGUI'

        default_gui_config['canvas'] = {'bg': 'white',
                                        'borderwidth': 0}

        default_gui_config['content_frame'] = {'bg': 'white',
                                                'borderwidth': 0}

        default_gui_config['data_elements'] = []

        default_gui_config['data_element_labelframe'] = dict()
        default_gui_config['data_element_title_format'] = None

        default_gui_config['data_sort_variables'] = None    # list with names of the variables for which the data can be sorted
                                                            # data will be sorted after the first variable

        default_gui_config['data_sort_direction'] = 'ascending'  # initial sort direction: ascending or descending

        default_gui_config['sort_options_frame'] = dict()

        default_gui_config['is_data_sort_variables_changeable'] = False
        default_gui_config['data_sort_variable_combobox'] = dict()
        default_gui_config['data_sort_variable_label'] = {'text': 'Sort variable:',
                                                          'anchor': 'nw'}

        default_gui_config['is_data_sort_direction_changeable'] = False
        default_gui_config['data_sort_direction_button'] = dict()
        default_gui_config['data_sort_direction_label'] = {'text': 'Sort direction:',
                                                           'anchor': 'nw'}

        default_gui_config['num_of_columns'] = None

        default_gui_config['data_labels'] = {'anchor': 'nw'}   #

        return default_gui_config


    def __init__(self, master=None, data=None, gui_config=None, **kwargs):
        '''
            Data must be a dictionary that holds the different data elements.
        '''

        super().__init__(master=master, gui_config=gui_config, **kwargs)

        if data is None:
            self.data = []
        else:
            self.data = data

        # Check if configuration is correct

        # either define number of cols or number of columns, but not both


        # variable holds all the elements that where created for each data entry
        self.elements = []

        # variables for sorting
        if self.gui_config['data_sort_variables']:
            self.data_sort_variable = self.gui_config['data_sort_variables'][0]
        else:
            self.data_sort_variable = None
        self.data_sort_direction = self.gui_config['data_sort_direction']

        # hold the bindings (i.e. functions that should be called) for the button click events on the displayed data entries
        self.event_button1_bindings = []
        self.event_button2_bindings = []
        self.event_button3_bindings = []

        self.create_gui()

        self.display_data(self.data)


    def create_gui(self):

        # make the canvas resizable
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # canvas is used to allow scrollbars, because the frame widget does not allow scrolling by itself
        self.canvas = tk.Canvas(self, **self.gui_config['canvas'])
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)

        # scroolbars
        self.scrollbar_y = AutoScrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar_y.grid(column=1, row=0, sticky=tk.NS)

        self.scrollbar_x = AutoScrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar_x.grid(column=0, row=1, sticky=tk.EW)


        if (self.gui_config['data_sort_variables'] and self.gui_config['is_data_sort_variables_changeable']) or self.gui_config['is_data_sort_direction_changeable']:
            # sort variable

            self.sort_options_frame = tk.Frame(master=self, **self.gui_config['sort_options_frame'])
            self.sort_options_frame.grid(row = 2, column=0, columnspan=2, sticky=tk.NSEW)
            self.sort_options_frame.columnconfigure(0, weight=1)

        if self.gui_config['data_sort_variables'] and self.gui_config['is_data_sort_variables_changeable']:
            self.data_sort_variable_label = tk.Label(master=self.sort_options_frame, **self.gui_config['data_sort_variable_label'])
            self.data_sort_variable_label.grid(column=1, row=0)

            self.data_sort_variable_combobox = ttk.Combobox(master=self.sort_options_frame, **self.gui_config['data_sort_variable_combobox'])
            self.data_sort_variable_combobox.grid(column=2, row=0, sticky=tk.NSEW)
            self.data_sort_variable_combobox.configure(values=self.gui_config['data_sort_variables'])
            self.data_sort_variable_combobox.current(0)
            self.data_sort_variable_combobox.bind('<<ComboboxSelected>>', self.on_data_sort_variable_combobox_change)

        if self.gui_config['is_data_sort_direction_changeable']:
            self.data_sort_direction_label = tk.Label(master=self.sort_options_frame, **self.gui_config['data_sort_direction_label'])
            self.data_sort_direction_label.grid(column=3, row=0)

            text = '>' if self.data_sort_direction == 'descending' else '<'
            self.data_sort_direction_button = tk.Button(master=self.sort_options_frame, text=text, command=self.on_data_sort_direction_button_press, **self.gui_config['data_sort_direction_button'])
            self.data_sort_direction_button.grid(column=4, row=0, sticky=tk.NSEW)
            self.data_sort_direction_button.bind('<<ComboboxSelected>>', self.on_data_sort_variable_combobox_change)

        self.canvas.config(yscrollcommand=self.scrollbar_y.set)
        self.canvas.config(xscrollcommand=self.scrollbar_x.set)

        # the frame that is used for the content
        self.content_frame = tk.Frame(self.canvas, **self.gui_config['content_frame'])

        # the frame gets added to the canvas
        self.canvas.create_window((0, 0), window=self.content_frame, anchor=tk.NW, tags="self.frame")

        self.content_frame.bind("<Configure>", self.on_content_frame_configure)


    def on_data_sort_variable_combobox_change(self, event):
        if self.data_sort_variable != self.data_sort_variable_combobox.get():
            self.data_sort_variable = self.data_sort_variable_combobox.get()
            self.sort_data_idxs()


    def on_data_sort_direction_button_press(self):
        if self.data_sort_direction == 'descending':
            self.data_sort_direction = 'ascending'
            text = '<'
        else:
            self.data_sort_direction = 'descending'
            text = '>'

        self.data_sort_direction_button.configure(text=text)

        self.sort_data_idxs()


    def on_content_frame_configure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


    def bind(self, sequence=None, func=None, add=None):

        if sequence == '<Button-1>':
            self.event_button1_bindings.append(func)
        elif sequence == '<Button-2>':
            self.event_button2_bindings.append(func)
        elif sequence == '<Button-3>':
            self.event_button3_bindings.append(func)
        else:
            super().bind(sequence=None, func=None, add=None)


    def display_data(self, data):

        self.data = data

        if self.elements:
            del(self.elements)
            self.elements = []

        if self.data is not None:

            # sorting of data according to one of the data fields
            self.sort_data_idxs()

            for data_idx, single_data in enumerate(self.data):
                pos = self.get_sorted_data_pos(data_idx)
                self.add_data_element(single_data, pos)


    def sort_data_idxs(self):

        if self.data_sort_variable:
            # create list of all the values for the sortvariable of all the data entries
            sortable_data = [data_entry[self.data_sort_variable] for data_entry in self.data]

            # sort them and save sorted indexes
            sorted_idxs = np.argsort(sortable_data)
            if self.data_sort_direction == 'descending':
                sorted_idxs = np.flip(sorted_idxs, axis=0)

        else:
            sorted_idxs = np.array(range(len(self.data)))

        self.sorted_data_idxs = sorted_idxs

        # resort existing elements
        for idx, (element, subelements) in enumerate(self.elements):
            pos = self.get_sorted_data_pos(idx)
            element.grid(row=pos[0], column=pos[1])


    def get_sorted_data_pos(self, data_idx):
        # find position of the data_entry
        sorted_idx = np.where(self.sorted_data_idxs == data_idx)[0][0]

        num_of_columns = self.gui_config['num_of_columns']

        if num_of_columns is None:
            # use ration of 16:9
            num_of_columns = max(1,int(math.ceil(math.sqrt(len(self.sorted_data_idxs) * 16 / 9))))

        if num_of_columns:
            row = int(sorted_idx / num_of_columns)
            col = sorted_idx % num_of_columns
        else:
            row = 0
            col = sorted_idx

        return row, col


    def add_data_element(self, data, pos):
        '''Adds for the given data an element to the main canvas.'''

        # create the base canvas where all other elements will be added

        element_frame = tk.LabelFrame(master=self.content_frame, **self.gui_config['data_element_labelframe'])
        element_frame.grid(row=pos[0], column=pos[1], sticky=tk.NW)

        element_frame.bind("<Button-1>", self.data_button_click)
        element_frame.bind("<Button-2>", self.data_button_click)
        element_frame.bind("<Button-3>", self.data_button_click)

        # set the title of the labelframe, if configured
        if self.gui_config['data_element_title_format']:
            title = self.gui_config['data_element_title_format'].format(**data)
            element_frame.configure(text=title)

        # add sub elements, depending on the configuration
        subelements = []
        cur_row = 0
        for element_config in self.gui_config['data_elements']:

            if 'label' in element_config and element_config['label'] is not None:
                label = tk.Label(master=element_frame, text=element_config['label'], **self.gui_config['data_labels'])
                label.grid(row=cur_row, column=0, sticky=tk.NW + tk.E)
                cur_row = cur_row + 1

                label.bind("<Button-1>", self.data_button_click)
                label.bind("<Button-2>", self.data_button_click)
                label.bind("<Button-3>", self.data_button_click)

            type = element_config['type']
            val = data[element_config['source_variable']]

            if isinstance(element_config['type'], str):
                # a known type is defined

                if 'format' in element_config:
                    val = element_config['format'].format(val)

                if type.lower() == 'label':
                    element = tk.Label(master=element_frame, text=val)
                elif type.lower() == 'message':
                    element = tk.Message(master=element_frame, text=val)
                else:
                    raise ValueError('Type {!r} is a unknown data element type! Please change the gui configuration.'.format(type))

                # if a gui_config is given then give it as input to the config
                if 'gui_config' in element_config:
                    element.configure(**element_config['gui_config'])

                element.grid(row=cur_row, column=0, sticky=tk.NW + tk.E)
                cur_row = cur_row + 1

            else:
                # a class is given
                element_class = element_config['type']
                init_params = {element_config['data_parameter_name']: val}

                # # if a gui_config is given then give it as input to the config
                # if 'gui_config' in element_config:
                #     init_params['gui_config'] = element_config['gui_config']

                element = element_class(master=element_frame, gui_config=element_config.get('gui_config'))
                element.grid(row=cur_row, column=0, sticky=tk.NW + tk.E)
                cur_row = cur_row + 1
                element.display_data(**init_params)

            element.bind("<Button-1>", self.data_button_click)
            element.bind("<Button-2>", self.data_button_click)
            element.bind("<Button-3>", self.data_button_click)

            subelements.append(element)

        self.elements.append((element_frame, subelements))


    def data_button_click(self, event):
        # identify the element which produced the event

        if (event.num == 1 and self.event_button1_bindings) \
           or (event.num == 2 and self.event_button2_bindings) \
           or (event.num == 3 and self.event_button3_bindings):

            # identify the id of the LabelFrame from which the event comes
            labelframe_id = None
            widget = event.widget
            content_frame_id = self.content_frame.winfo_id()
            while widget.master is not None:
                if widget.master.winfo_id() == content_frame_id:
                    labelframe_id = widget.winfo_id()
                    break
                widget = widget.master

            # identify the idx of the labelframe from where the event comes
            element_idx = None
            if labelframe_id is not None:
                for idx, (elem, subelements) in enumerate(self.elements):
                    if elem.winfo_id() == labelframe_id:
                        element_idx = idx
                        break

            event.data_idx = element_idx

            if event.num == 1:
                for func in self.event_button1_bindings:
                    func(event)
            elif event.num == 2:
                for func in self.event_button2_bindings:
                    func(event)
            elif event.num == 3:
                for func in self.event_button3_bindings:
                    func(event)

