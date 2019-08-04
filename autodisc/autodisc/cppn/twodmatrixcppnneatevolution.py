import autodisc as ad
from autodisc.cppn.selfconnectiongenome import SelfConnectionGenome
import neat
import copy
import random

class TwoDMatrixCCPNNEATEvolution:

    @staticmethod
    def default_config():
        def_config = ad.Config()
        def_config.neat_config_file = 'neat.cfg'
        def_config.matrix_size = (100, 100)
        def_config.cppn_input_borders = ((-2,2), (-2,2))
        def_config.recurrent_net_repetitions = 4 # number of iterations a recurrent nerual network is executed
        def_config.n_generations = 1
        def_config.is_verbose = False
        def_config.is_extra_fitness_data = False # does the fitness function returns extra data that should be saved
        def_config.is_pytorch = True
        def_config.keep_results = 'none'  # 'none', 'all_gen', 'last_gen'
        def_config.fitness_function = lambda matrix, genome: 0
        def_config.fitness_function_param = None

        return def_config


    def __init__(self, init_population=None, config=None, **kwargs):
        '''

        Configuration:

            keep_results: Defines if the results of the exploration should be kept.
                          Options:
                                - 'none' (default)
                                - 'all_gen': keep results from all generations
                                - 'last_gen': keep results from last generation

            fitness_function: Function pointer to fitness function. Form: [fitness (,data)] = fitness_function(matrix, genome (, fitness_function_param)).
                              Set config.is_extra_fitness_data to True if extra data is returned.
                
            fitness_function_param: Optional parameter for the fitness function. Can be used to configure the function.

        :param init_population: List of genomes that are used as initial population.
                                If less are given than pop_size, the others are randomly generated. If more are given, then the elements will be randomly chosen.

        :param config: Configuration
        '''
        self.config = ad.config.set_default_config(kwargs, config, TwoDMatrixCCPNNEATEvolution.default_config())

        if init_population is not None and not isinstance(init_population, list):
            init_population = [init_population]

        self.generation = -1

        self.neat_config = neat.Config(SelfConnectionGenome,
                                       neat.DefaultReproduction,
                                       neat.DefaultSpeciesSet,
                                       neat.DefaultStagnation,
                                       self.config.neat_config_file)

        # add userdefined activation functions
        self.neat_config.genome_config.add_activation('delphineat_gauss', ad.cppn.activations.delphineat_gauss_activation)
        self.neat_config.genome_config.add_activation('delphineat_sigmoid', ad.cppn.activations.delphineat_sigmoid_activation)

        # regular neat evolution can not handle population size of 1
        if self.neat_config.pop_size == 1:

            self.is_single_cppn_evolution = True

            # population is a single genome
            if init_population is None:
                self.population = self.neat_config.genome_type(0)
                self.population.configure_new(self.neat_config.genome_config)
            else:
                self.population = copy.deepcopy(random.choice(init_population))

            # hold best genome here
            if self.neat_config.fitness_criterion == 'max':
                self.fitness_criterion = lambda g1, g2: g2 if g2.fitness > g1.fitness else g1
            elif self.neat_config.fitness_criterion == 'min':
                self.fitness_criterion = lambda g1, g2: g2 if g2.fitness < g1.fitness else g1
            else:
                raise ValueError('Usupported fitness criterion {!r}! Evolutions with population size of 1 supports only \'min\' or \'max\'.'.format(self.neat_config.fitness_criterion))

            self.best_genome = self.population

        else:

            self.is_single_cppn_evolution = False

            if init_population is None:
                self.population = neat.Population(self.neat_config)
            else:

                # if more elements in init list than population size, use random order
                if len(init_population) <= self.neat_config.pop_size:
                    init_population_tmp = init_population
                else:
                    init_population_tmp = random.shuffle(init_population.copy())

                population = {}

                # if there are randomly generated genomes, use for their keys new keys
                next_key = max([genome.key for genome in init_population_tmp]) + 1
                for idx in range(self.neat_config.pop_size):

                    if idx < len(init_population_tmp):
                        genome = copy.deepcopy(init_population_tmp[idx])
                    else:
                        genome = self.neat_config.genome_type(0)
                        genome.configure_new(self.neat_config.genome_config)
                        genome.key = next_key

                        next_key += 1

                    population[genome.key] = genome

                species = config.species_set_type(self.neat_config.species_set_config, neat.reporting.ReporterSet())
                species.speciate(neat.config, population, 0)

                initial_state = (population, species, 0)
                self.population = neat.Population(self.neat_config, initial_state)

                for genome_key in population.keys():
                    self.population.population.reproduction.ancestors[genome_key] = tuple()


        if self.config['is_verbose'] and not self.is_single_cppn_evolution:
            # Add a stdout reporter to show progress in the terminal.
            self.population.add_reporter(neat.StdOutReporter(True))
            self.statistics_reporter = neat.StatisticsReporter()
            self.population.add_reporter(self.statistics_reporter)

        # save some config parametrs as variables for increased performance
        self.matrix_size =  self.config.matrix_size
        self.recurrent_net_repetitions = self.config.recurrent_net_repetitions
        self.is_extra_fitness_data = self.config.is_extra_fitness_data

        if self.config.keep_results.lower() == 'none':
            self.is_keep_all_gen_results = False
            self.is_keep_last_gen_results = False
        elif self.config.keep_results.lower() == 'all_gen':
            self.is_keep_all_gen_results = True
            self.is_keep_last_gen_results = False
        elif self.config.keep_results.lower() == 'last_gen':
            self.is_keep_all_gen_results = False
            self.is_keep_last_gen_results = True
        else:
            raise ValueError('Unknown keep_results configuration {!r}!. Allowed values: none, all_gen, last_gen', self.config.keep_results)

        if self.is_keep_last_gen_results or self.is_keep_all_gen_results:
            self.results = dict()
        else:
            self.results = None

        self.net_input = ad.cppn.helper.create_image_cppn_input(self.matrix_size, input_borders=self.config.cppn_input_borders)


    def get_best_genome(self):
        if self.is_single_cppn_evolution:
            return self.best_genome
        else:
            return self.population.best_genome


    def get_best_genome_last_generation(self):
        if self.is_single_cppn_evolution:
            return self.population
        else:
            max_fitness = float('-inf')
            max_fitness_idx = None

            for idx, genome in enumerate(self.population):
                if genome.fitness > max_fitness:
                    max_fitness = genome.fitness
                    max_fitness_idx = idx

            return self.population[max_fitness_idx]


    def get_best_matrix(self):
        # TODO: save best matrix immediately after it was generated, so that this computation can be avoided
        best_genome = self.get_best_genome()
        return self.genome_to_matrix(best_genome, self.neat_config)


    def get_best_matrix_last_generation(self):
        # TODO: save best matrix immediately after it was generated, so that this computation can be avoided
        best_genome = self.get_best_genome_last_generation()
        return self.genome_to_matrix(best_genome, self.neat_config)


    def do_evolution(self, n_generations=None):

        self.generation = -1

        # use default number from config if nothing is given
        if n_generations is None:
            n_generations = self.config.n_generations

        for _ in range(n_generations):
            self.do_next_generation()


    def do_next_generation(self):

        self.generation += 1

        if self.is_single_cppn_evolution:

            # use the initialized genome for the first generation
            if self.generation > 0:
                self.population = copy.deepcopy(self.population)
                self.population.key = self.generation
                self.population.mutate(self.neat_config.genome_config)

            self.eval_population_fitness([(self.population.key, self.population)], self.neat_config)

            # update best genome according to the fitness criteria
            self.best_genome = self.fitness_criterion(self.best_genome, self.population)
        else:
            self.population.run(self.eval_population_fitness, 1)

        if self.is_keep_last_gen_results and self.generation > 0:
            del(self.results[self.generation-1])


    def eval_population_fitness(self, genomes, neat_config):

        if self.is_keep_last_gen_results or self.is_keep_all_gen_results:
            results = []

        for (genome_id, genome) in genomes:

            mat = self.genome_to_matrix(genome, neat_config)

            if (self.is_keep_last_gen_results or self.is_keep_all_gen_results) and self.is_extra_fitness_data:

                if self.config.fitness_function_param is None:
                    [genome.fitness, extra_data] = self.config.fitness_function(mat, genome)
                else:
                    [genome.fitness, extra_data] = self.config.fitness_function(mat, genome, self.config.fitness_function_param)

            else:

                if self.config.fitness_function_param is None:
                    genome.fitness = self.config.fitness_function(mat, genome)
                else:
                    genome.fitness = self.config.fitness_function(mat, genome, self.config.fitness_function_param)

            if self.is_keep_last_gen_results or self.is_keep_all_gen_results:
                result = dict()
                result['id'] = genome.key
                result['genome'] = genome
                result['matrix'] = mat
                result['fitness'] = genome.fitness

                if self.is_extra_fitness_data:
                    result['data'] = extra_data

                results.append(result)

        if self.is_keep_last_gen_results or self.is_keep_all_gen_results:
            self.results[self.generation] = results


    def genome_to_matrix(self, genome, neat_config):

        if self.config['is_pytorch']:

            if neat_config.genome_config.feed_forward:
                raise NotImplementedError('Feedforward networks for pytorch are not implemented!')
            else:
                net = ad.cppn.pytorchcppn.RecurrentCPPN.create_from_genome(genome, neat_config)
                net_output = net.activate(self.net_input, self.recurrent_net_repetitions)
        else:

            if neat_config.genome_config.feed_forward:
                net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
                net_output = ad.cppn.helper.calc_neat_forward_image_cppn_output(net, self.net_input)


            else:
                net = neat.nn.RecurrentNetwork.create(genome, neat_config)
                net_output = ad.cppn.helper.calc_neat_recurrent_image_cppn_output(net, self.net_input, self.recurrent_net_repetitions)

        mat = ad.cppn.helper.postprocess_image_cppn_output(self.matrix_size, net_output)

        return mat