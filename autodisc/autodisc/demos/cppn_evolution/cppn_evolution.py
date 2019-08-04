#!/usr/bin/env python

from autodisc.cppn.twodmatrixcppnneatevolution import TwoDMatrixCCPNNEATEvolution
from autodisc.cppn.neatcppngui import NeatCPPNEvolutionGUI


def fitness_function(image, genome):
    return 0

evo_config = TwoDMatrixCCPNNEATEvolution.default_config()
evo_config['is_verbose'] = True
evo_config['keep_results'] = 'all_gen'
evo_config['matrix_size'] = (128, 128)
evo_config['is_pytorch'] = True

evo = TwoDMatrixCCPNNEATEvolution(fitness_function=fitness_function, config=evo_config)
evo.do_next_generation() # generate first generation

# run the gui

gui_config = NeatCPPNEvolutionGUI.default_gui_config()
gui_config['dialog']['geometry'] = '1024x768'

gui = NeatCPPNEvolutionGUI(evolution=evo, gui_config=gui_config)
gui.run()
