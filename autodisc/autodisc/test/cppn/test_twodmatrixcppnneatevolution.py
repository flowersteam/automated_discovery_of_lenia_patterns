import autodisc as ad
import os

def test_twodmatrixcppnneatevolution():

    dir_path = os.path.dirname(os.path.realpath(__file__))

    ################################################################
    # normal evolution

    evo_config = ad.cppn.TwoDMatrixCCPNNEATEvolution.default_config()

    evo_config.neat_config_file = os.path.join(dir_path, 'test_neat.cfg')
    evo_config.is_verbose = False
    evo_config.keep_results = 'all_gen'
    evo_config.matrix_size = (50, 50)
    evo_config.is_pytorch = True

    evo = ad.cppn.TwoDMatrixCCPNNEATEvolution(fitness_function=lambda mat, genome: 0, config=evo_config)
    evo.do_next_generation()  # generate first generation

    assert len(evo.results) == 1
    assert len(evo.results[0]) == evo.neat_config.pop_size


    evo.do_next_generation()  # generate first generation

    assert len(evo.results) == 2
    assert len(evo.results[0]) == evo.neat_config.pop_size
    assert len(evo.results[1]) == evo.neat_config.pop_size


    ################################################################
    # keep ony last generation

    evo_config = ad.cppn.TwoDMatrixCCPNNEATEvolution.default_config()

    evo_config.neat_config_file = os.path.join(dir_path, 'test_neat.cfg')
    evo_config.is_verbose = False
    evo_config.keep_results = 'last_gen'
    evo_config.matrix_size = (50, 50)
    evo_config.is_pytorch = True

    evo = ad.cppn.TwoDMatrixCCPNNEATEvolution(fitness_function=lambda mat, genome: 0, config=evo_config)
    evo.do_next_generation()  # generate first generation

    assert len(evo.results) == 1
    assert len(evo.results[0]) == evo.neat_config.pop_size

    evo.do_next_generation()  # generate first generation

    assert len(evo.results) == 1
    assert len(evo.results[1]) == evo.neat_config.pop_size


    ################################################################
    # population_size == 1

    evo_config = ad.cppn.TwoDMatrixCCPNNEATEvolution.default_config()

    evo_config.neat_config_file = os.path.join(dir_path, 'test_neat_single.cfg')
    evo_config.is_verbose = False
    evo_config.keep_results = 'all_gen'
    evo_config.matrix_size = (50, 50)
    evo_config.is_pytorch = True

    evo = ad.cppn.TwoDMatrixCCPNNEATEvolution(fitness_function=lambda mat, genome: 0, config=evo_config)
    evo.do_next_generation()  # generate first generation

    assert len(evo.results) == 1
    assert len(evo.results[0]) == evo.neat_config.pop_size

    evo.do_next_generation()  # generate first generation

    assert len(evo.results) == 2
    assert len(evo.results[0]) == evo.neat_config.pop_size
    assert len(evo.results[1]) == evo.neat_config.pop_size

