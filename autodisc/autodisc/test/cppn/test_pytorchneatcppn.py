import autodisc as ad
from autodisc.cppn.selfconnectiongenome import SelfConnectionGenome
import neat

import os
import numpy as np
import random
import torch

def test_RecurrentCPPNNet():

    ###########################################################
    # Create a number of random networks and test if the neat rnn and the pytorch rnn generate the same output for each.

    num_networks = 10
    recurrent_net_repetitions = 4

    # load test configuration
    config_path = os.path.join(os.path.dirname(__file__), 'test_neat.cfg')
    neat_config = neat.Config(
        SelfConnectionGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # add custom activation functions
    neat_config.genome_config.add_activation('delphineat_gauss', ad.cppn.activations.delphineat_gauss_activation)
    neat_config.genome_config.add_activation('delphineat_sigmoid', ad.cppn.activations.delphineat_sigmoid_activation)

    # create the genome
    cppn_input = ad.cppn.helper.create_image_cppn_input((50, 50))

    for net_idx in range(num_networks):

        # set seed to make result reproducible
        random.seed(net_idx)

        # create a network
        genome = neat_config.genome_type(net_idx)
        genome.configure_new(neat_config.genome_config)

        # test its outputs
        neat_net = neat.nn.RecurrentNetwork.create(genome, neat_config)
        neat_output = ad.cppn.helper.calc_neat_recurrent_image_cppn_output(neat_net, cppn_input, recurrent_net_repetitions)

        pytorch_net = ad.cppn.pytorchcppn.RecurrentCPPN.create_from_genome(genome, neat_config, dtype=torch.float64)
        pytorch_output = pytorch_net.activate(cppn_input, recurrent_net_repetitions)
        pytorch_output = pytorch_output.reshape((len(cppn_input),))

        assert np.allclose(neat_output, pytorch_output)



    ###########################################################
    # Create a network which has a node that has no intput nodes

    genome = neat_config.genome_type(net_idx)
    genome.configure_new(neat_config.genome_config)

    # change network structure
    trg_node = list(genome.nodes.keys())[-1]

    delete_keys = []

    for key, _ in genome.connections.items():
        if key[1] == trg_node:
            delete_keys.append(key)

    for key in delete_keys:
        del genome.connections[key]

    # test its outputs
    neat_net = neat.nn.RecurrentNetwork.create(genome, neat_config)
    neat_output = ad.cppn.helper.calc_neat_recurrent_image_cppn_output(neat_net, cppn_input, recurrent_net_repetitions)

    pytorch_net = ad.cppn.pytorchcppn.RecurrentCPPN.create_from_genome(genome, neat_config, dtype=torch.float64)
    pytorch_output = pytorch_net.activate(cppn_input, recurrent_net_repetitions)
    pytorch_output = pytorch_output.reshape((len(cppn_input),))

    assert np.allclose(neat_output, pytorch_output)


    ###############################################################
    # network without connections

    genome = neat_config.genome_type(net_idx)
    genome.configure_new(neat_config.genome_config)

    # change network structure
    genome.connections.clear()

    # test its outputs
    neat_net = neat.nn.RecurrentNetwork.create(genome, neat_config)
    neat_output = ad.cppn.helper.calc_neat_recurrent_image_cppn_output(neat_net, cppn_input, recurrent_net_repetitions)

    pytorch_net = ad.cppn.pytorchcppn.RecurrentCPPN.create_from_genome(genome, neat_config, dtype=torch.float64)
    pytorch_output = pytorch_net.activate(cppn_input, recurrent_net_repetitions)
    pytorch_output = pytorch_output.reshape((len(cppn_input),))

    assert np.allclose(neat_output, pytorch_output)




