import autodisc as ad

class RandomExplorer(ad.core.Explorer):
    '''Performs random explorations of a system.'''

    @staticmethod
    def default_config():
        def_config = ad.core.Explorer.default_config()

        def_config.num_of_steps = 100
        def_config.default_run_parameters = ad.helper.data.AttrDict()
        def_config.explore_parameters = ad.helper.data.AttrDict()

        return def_config


    def run(self, num_of_runs=100, verbose=True):

        if verbose:
            ad.gui.print_progress_bar(0, num_of_runs, 'Explorations: ')

        for run_idx in range(num_of_runs):

            if run_idx not in self.data:

                # set a new seed for the new run depending on the init_seed

                if self.config.seed is not None:
                    seed = self.config.seed * 1000 + run_idx
                    self.random.seed(seed)
                else:
                    seed = None

                # sample parameters and set them
                run_parameters = self.config.default_run_parameters.copy()
                for param_name, param_properties in self.config.explore_parameters.items():
                    run_parameters[param_name] = ad.helper.sampling.sample_value(self.random, param_properties) # sample properties

                # run the experiment
                observations, statistics = self.system.run(run_parameters=run_parameters, stop_conditions=self.config.num_of_steps)

                self.data.add_run_data(id=run_idx,
                                       seed=seed,
                                       run_parameters=run_parameters,
                                       observations=observations,
                                       statistics=statistics)

            if verbose:
                ad.gui.print_progress_bar(run_idx + 1, num_of_runs, 'Explorations: ')