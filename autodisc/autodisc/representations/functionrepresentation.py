import autodisc as ad
import importlib


class FunctionRepresentation(ad.core.Representation):

    @staticmethod
    def default_config():
        default_config = ad.core.Representation.default_config()
        default_config.function = None
        #default_config.config : if defined then this is given to the function as extra parameter

        return default_config


    def calc(self, observations, statistics):

        if isinstance(self.config.function, str):
            # load external function
            module_name = '.'.join(self.config.function.split('.')[0:-1])
            function_name = self.config.function.split('.')[-1]

            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
        else:
            function = self.config.function

        if 'config' in self.config:
            representation = function(observations, statistics, self.config.config)
        else:
            representation = function(observations, statistics)

        return representation

