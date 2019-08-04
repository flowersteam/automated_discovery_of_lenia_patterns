import autodisc as ad
from autodisc.gui import ArrayToImageGUI, BaseFrame, DataViewerGUI, ImageViewerGUI
try:
    import tkinter as tk
except:
    import Tkinter as tk
import os
import copy
import warnings
import sys

# graphviz is only optionally, if not installed, then the networks can not be displayed
try:
    import graphviz
except:
    pass

from PIL import Image


class NeatCPPNEvolutionGUI(BaseFrame):

    @staticmethod
    def default_gui_config():
        default_gui_config = BaseFrame.default_gui_config()

        default_gui_config['dialog']['title'] = 'CPPN NEAT GUI'
        default_gui_config['dataviewer'] = DataViewerGUI.default_gui_config()

        default_gui_config['dataviewer']['data_element_labelframe']['bg'] = 'white'

        default_gui_config['dataviewer']['data_labels']['bg'] = 'white'

        default_gui_config['dataviewer']['data_elements'].append({'type': ArrayToImageGUI,
                                                                  'data_parameter_name': 'image_data',
                                                                  'source_variable': 'matrix',
                                                                  'gui_config': {'pixel_size': 1}})

        default_gui_config['dataviewer']['data_elements'].append({'type': 'Label',
                                                                  'format': '{:.3f}',
                                                                  'source_variable': 'fitness',
                                                                  'label': 'Fitness:',
                                                                  'gui_config': {'bg': 'white'}})

        default_gui_config['dataviewer']['data_element_title_format'] = 'ID: {id}'

        default_gui_config['dataviewer']['is_data_sort_variables_changeable'] = True
        default_gui_config['dataviewer']['data_sort_variables'] = ['fitness', 'id']

        default_gui_config['dataviewer']['is_data_sort_direction_changeable'] = True
        default_gui_config['dataviewer']['data_sort_direction'] = 'descending'

        default_gui_config['networkviewer'] = NEATCPPNNetworkViewerGUI.default_gui_config()
        default_gui_config['networkviewer']['dialog']['is_transient'] = True # don't use individual task bar items for the cppn network viewers

        default_gui_config['imageviewer'] = NEATCPPNNetworkViewerGUI.default_gui_config()
        default_gui_config['imageviewer']['dialog']['is_transient'] = True # don't use individual task bar items for the cppn network viewers
        default_gui_config['imageviewer']['pixel_size'] = 4  # don't use individual task bar items for the cppn network viewers


        default_gui_config['label_generation'] = dict()
        default_gui_config['label_generation']['anchor'] = tk.NW

        default_gui_config['button_next_gen'] = dict()
        default_gui_config['button_next_gen']['text'] = 'Evolve next generation...'

        default_gui_config['button_goto_previous_gen'] = {'text': '<'}
        default_gui_config['button_goto_next_gen'] = {'text': '>'}

        return default_gui_config


    def __init__(self, master=None, is_dialog=False, evolution=None, gui_config=None, **kwargs):

        assert evolution is not None
        self.evolution = evolution

        super().__init__(master=master, is_dialog=is_dialog, gui_config=gui_config, **kwargs)

        self.create_gui()

        self.dislayed_gen = None
        self.display_generation_data()


    def create_gui(self):

        self.columnconfigure(0, weight=1)

        # if between generations should be switched then make the default columnspan 2 to allow the two switch buttons on the bottom of the view
        if self.evolution.is_keep_all_gen_results:
            columnspan = 2
        else:
            columnspan = 1

        self.label_generation = tk.Label(master=self, **self.gui_config['label_generation'])
        self.label_generation.grid(column=0, row=0, columnspan=columnspan, sticky=tk.NSEW)

        # add data viewer
        self.data_viewer = DataViewerGUI(master=self, gui_config=self.gui_config['dataviewer'])
        self.data_viewer.grid(column=0, row=1, columnspan=columnspan, sticky=tk.NSEW)
        self.data_viewer.bind('<Button-1>', self.on_data_click)
        self.data_viewer.bind('<Button-3>', self.on_data_click)

        self.rowconfigure(1, weight=1)

        if self.evolution.is_keep_all_gen_results:
            self.button_goto_previous_gen = tk.Button(master=self, command=self.goto_previous_generation, **self.gui_config['button_goto_previous_gen'])
            self.button_goto_previous_gen.grid(column=0, row=2, columnspan=1, sticky=tk.NSEW)

            self.button_goto_next_gen = tk.Button(master=self, command=self.goto_next_generation, **self.gui_config['button_goto_next_gen'])
            self.button_goto_next_gen.grid(column=1, row=2, columnspan = 1, sticky=tk.NSEW)

            self.columnconfigure(0, weight=1)
            self.columnconfigure(1, weight=1)


        #  add button for next generation
        self.button_next_gen = tk.Button(master=self, command=self.do_next_generation, **self.gui_config['button_next_gen'])
        self.button_next_gen.grid(column=0, row=3, columnspan=columnspan, sticky=tk.NSEW)


    def on_data_click(self, event):

        if event.num == 1:
            # leftclick

            # create a new dialog that shows the cppn network
            gui = NEATCPPNNetworkViewerGUI(master=self,
                                           is_dialog=True,
                                           gui_config=self.gui_config['networkviewer'])

            gui.display_data(network_genome=self.evolution.results[self.dislayed_gen][event.data_idx]['genome'],
                             neat_config=self.evolution.neat_config,
                             info_string='Generation: {}, ID: {}'.format(self.dislayed_gen, self.evolution.results[self.dislayed_gen][event.data_idx]['id']))

        elif event.num == 3:
            # rightclick
            gui = ArrayToImageGUI(master=self,
                                  is_dialog=True,
                                  gui_config=self.gui_config['imageviewer'])

            gui.display_data(image_data=self.evolution.results[self.dislayed_gen][event.data_idx]['matrix'],
                             image_names='Generation: {}, ID: {}'.format(self.dislayed_gen, self.evolution.results[self.dislayed_gen][event.data_idx]['id']))



    def do_next_generation(self):
        self.evolution.do_next_generation()
        self.display_generation_data()


    def goto_previous_generation(self):
        if self.dislayed_gen is not None and self.dislayed_gen > 0:
            self.display_generation_data(generation=self.dislayed_gen-1)


    def goto_next_generation(self):
        if self.dislayed_gen is not None and self.dislayed_gen < self.evolution.generation:
            self.display_generation_data(generation=self.dislayed_gen+1)


    def display_generation_data(self, generation=None):

        if generation is None or generation == -1:
            generation = self.evolution.generation

        if generation is not None:

            self.dislayed_gen = generation

            self.label_generation.configure(text='Generation: {}'.format(self.dislayed_gen))

            # update the data from the data viewer
            self.data_viewer.display_data(self.evolution.results[self.dislayed_gen])

        elif self.evolution.generation is None:
            self.dislayed_gen = None
            self.label_generation.configure(text='Initialize first generation by clicking on \'Next Generation\'.')
        else:
            self.dislayed_gen = None
            self.label_generation.configure(text='')


    def run(self):
        self.master.mainloop()


class NEATCPPNNetworkViewerGUI(BaseFrame):

    @staticmethod
    def default_gui_config():
        default_gui_config = BaseFrame.default_gui_config()

        default_gui_config['dialog']['title'] = 'CPPN Network Viewer'

        default_gui_config['network_visualization'] = default_draw_network_visualization_config()

        default_gui_config['frame']['bg'] = 'green'

        default_gui_config['image_viewer'] = ImageViewerGUI.default_gui_config()
        default_gui_config['image_viewer']['bg'] = 'red'

        return default_gui_config


    def __init__(self, master=None, is_dialog=False, neat_config=None, network_genome=None, info_string=None, gui_config=None, **kwargs):

        self.network_genome = None
        self.neat_config = None
        self.info_string = None

        super().__init__(master=master, is_dialog=is_dialog, gui_config=gui_config, **kwargs)

        self.create_gui()


    def create_gui(self):

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # make a canvas in which the image is displayed
        self.image_viewer = ImageViewerGUI(master=self, gui_config=self.gui_config['image_viewer'])
        self.image_viewer.grid(row=0, column=0, sticky=tk.NSEW)


    def display_data(self, network_genome=None, neat_config=None, info_string=None):

        self.network_genome = network_genome
        self.neat_config = neat_config
        self.info_string = info_string

        if 'graphviz' in sys.modules:

            # creates a temporary png file that is afterwards deleted
            # TODO: check if dot.pipe() and Image.frombytes() methods can be used to avoid the temporary file
            filename = './tmp'
            fileformat = 'png'

            draw_network(neat_config=self.neat_config, genome=self.network_genome, filename=filename, fmt=fileformat, visualization_config=self.gui_config['network_visualization'])
            image = Image.open(filename + '.' + fileformat)
            os.remove(filename + '.' + fileformat)

            info = self.info_string
        else:
            image = None
            info = 'This display is not available due to a missing optional dependency (graphviz)!'

        self.image_viewer.display_data(image_data=image, image_names=info)


def default_draw_network_visualization_config():
    def_visualization_config = dict()

    def_visualization_config['is_show_disabled'] = True

    def_visualization_config['is_prune_unused'] = False

    def_visualization_config['activation_abrev'] = {'delphineat_sigmoid': 'd_sigm',
                                                    'delphineat_gauss': 'd_gauss'}

    def_visualization_config['aggregation_abrev'] = {'product': 'prod'}

    def_visualization_config['input_label_format'] = '{name}'
    def_visualization_config['node_label_format'] = '{name}\nagg: {aggregation} | act: {activation}\nb: {bias:.2f} | r: {response:.2f}'
    def_visualization_config['output_label_format'] = None

    def_visualization_config['default_neuron_attributes'] = {'shape': 'ellipse',
                                                             'fontsize': '10',
                                                             'height': '0.2',
                                                             'width': '0.2',
                                                             'style': 'filled',
                                                             'fillcolor': 'white'}

    def_visualization_config['input_neuron_attributes'] = {'style': 'filled',
                                                           'shape': 'box',
                                                           'fillcolor': 'lightgray'}

    def_visualization_config['output_neuron_attributes'] = {'style': 'filled',
                                                            'fillcolor': 'lightblue'}

    def_visualization_config['default_synqapse_attributes'] = {'fontsize': '10',
                                                               'penwidth': '1'}

    def_visualization_config['synapse_pos_negative_colors'] = ('blue', 'red')
    def_visualization_config['synapse_active_deactive_style'] = ('solid', 'dotted')

    def_visualization_config['synapse_width_coeff'] = 1

    def_visualization_config['synqapse_label_format'] = '{:.2f}'

    return def_visualization_config


def draw_network(neat_config, genome, view=False, filename=None, node_names=None, fmt='png', visualization_config=None):

    visualization_config = ad.helper.data.set_dict_default_values(visualization_config, default_draw_network_visualization_config())

    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)!")
        return

    def get_node_label(node_id, type='default'):

        node_info_dict = dict()

        if type == 'input':
            node_info_dict['name'] = node_names.get(node_id, str(node_id))

        else:
            node = genome.nodes[node_id]

            node_info_dict['name'] = node_names.get(node_id, str(node_id))
            node_info_dict['activation'] = visualization_config['activation_abrev'].get(node.activation, node.activation)
            node_info_dict['aggregation'] = visualization_config['aggregation_abrev'].get(node.aggregation, node.aggregation)
            node_info_dict['bias'] = node.bias
            node_info_dict['response'] = node.response

        label_format = visualization_config['node_label_format']

        if type == 'input' and visualization_config['input_label_format'] is not None:
            label_format = visualization_config['input_label_format']

        if type == 'output' and visualization_config['output_label_format'] is not None:
            label_format = visualization_config['output_label_format']

        return label_format.format(**node_info_dict)


    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    dot = graphviz.Digraph(format=fmt, node_attr=visualization_config['default_neuron_attributes'])

    # put nodes in respective subgraphs to have a nicer ordering
    input_subgraph = graphviz.Digraph('input_subgraph')
    input_subgraph.graph_attr.update(rank='same')

    hidden_subgraph = graphviz.Digraph('hidden_subgraph')

    output_subgraph = graphviz.Digraph('output_subgraph')
    output_subgraph.graph_attr.update(rank='same')

    inputs = set()
    prev_name = None
    for k in neat_config.genome_config.input_keys:
        inputs.add(k)

        name = node_names.get(k, str(k))
        label = get_node_label(k, 'input')
        node_attrs = visualization_config['input_neuron_attributes']

        input_subgraph.node(name, label=label, _attributes=node_attrs)

        # add insible edges to the inputs to order them as defined in the genome
        if prev_name is not None:
            input_subgraph.edge(prev_name, name, style='invis')
        prev_name = name

    outputs = set()
    prev_name = None
    for k in neat_config.genome_config.output_keys:
        outputs.add(k)

        name = node_names.get(k, str(k))
        label = get_node_label(k, 'output')
        node_attrs = visualization_config['output_neuron_attributes']

        output_subgraph.node(name, label=label, _attributes=node_attrs)

        # add insible edges to the inputs to order them as defined in the genome
        if prev_name is not None:
            output_subgraph.edge(prev_name, name, style='invis')
        prev_name = name

    if visualization_config['is_prune_unused']:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or visualization_config['is_show_disabled']:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        name = str(n)
        label = get_node_label(n)

        hidden_subgraph.node(name, label=label)

    for cg in genome.connections.values():
        if cg.enabled or visualization_config['is_show_disabled']:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))

            label = visualization_config['synqapse_label_format'].format(cg.weight)

            # get default attributes, and set some attributes depending on the synapse properties
            attributes = visualization_config['default_synqapse_attributes'].copy()

            if visualization_config['synapse_active_deactive_style'] is not None:
                attributes['style'] = visualization_config['synapse_active_deactive_style'][0] if cg.enabled else visualization_config['synapse_active_deactive_style'][1]

            if visualization_config['synapse_pos_negative_colors'] is not None:
                attributes['color'] = visualization_config['synapse_pos_negative_colors'][0] if cg.weight > 0 else visualization_config['synapse_pos_negative_colors'][1]

            if visualization_config['synapse_width_coeff'] is not None:
                attributes['penwidth'] = str(0.1 + abs(cg.weight * visualization_config['synapse_width_coeff']))

            dot.edge(a, b, label=label, _attributes=attributes)

    dot.subgraph(input_subgraph)
    dot.subgraph(hidden_subgraph)
    dot.subgraph(output_subgraph)

    dot.render(filename, view=view, cleanup=True)

    return dot


# if __name__ == '__main__':
#
#     gui_config = NeatCPPNEvolutionGUI.get_default_gui_config()
#
#     gui = NeatCPPNEvolutionGUI(gui_config=gui_config)
#     gui.run()