from autodisc.helper.data import AttrDict
from autodisc.helper.data import set_dict_default_values

class Config(AttrDict):

    def set_default_values(self, *args):

        args = list(args)
        for idx in range(len(args)):
            if args[idx] is None:
                args[idx] = AttrDict()
            elif not isinstance(args[idx], AttrDict):
                args[idx] = AttrDict.fromDict(args[idx])

        set_dict_default_values(self, *args, is_copy=False)



def set_default_config(*args, is_copy=True):

    args = list(args)
    for idx in range(len(args)):
        if args[idx] is None:
            args[idx] = AttrDict()
        elif not isinstance(args[idx], AttrDict):
            args[idx] = AttrDict.fromDict(args[idx])

    return set_dict_default_values(*args, is_copy=is_copy)








