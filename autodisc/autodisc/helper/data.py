import collections
import base64
import json
import numpy as np

from collections import defaultdict
from six import u, iteritems, iterkeys  # pylint: disable=unused-import


class AttrDict(dict):
    """ A dictionary that provides attribute-style access.
        >>> b = AttrDict()
        >>> b.hello = 'world'
        >>> b.hello
        'world'
        >>> b['hello'] += "!"
        >>> b.hello
        'world!'
        >>> b.foo = AttrDict(lol=True)
        >>> b.foo.lol
        True
        >>> b.foo is b['foo']
        True
        A Munch is a subclass of dict; it supports all the methods a dict does...
        >>> sorted(b.keys())
        ['foo', 'hello']
        Including update()...
        >>> b.update({ 'ponies': 'are pretty!' }, hello=42)
        >>> print (repr(b))
        Munch({'ponies': 'are pretty!', 'foo': Munch({'lol': True}), 'hello': 42})
        As well as iteration...
        >>> sorted([ (k,b[k]) for k in b ])
        [('foo', Munch({'lol': True})), ('hello', 42), ('ponies', 'are pretty!')]
        And "splats".
        >>> "The {knights} who say {ni}!".format(**AttrDict(knights='lolcats', ni='can haz'))
        'The lolcats who say can haz!'
        See unmunchify/Munch.toDict, munchify/Munch.fromDict for notes about conversion.
    """


    # only called if k not found in normal places
    def __getattr__(self, k):
        """ Gets key if it exists, otherwise throws AttributeError.
            nb. __getattr__ is only called if key is not found in normal places.
            >>> b = AttrDict(bar='baz', lol={})
            >>> b.foo
            Traceback (most recent call last):
                ...
            AttributeError: foo
            >>> b.bar
            'baz'
            >>> getattr(b, 'bar')
            'baz'
            >>> b['bar']
            'baz'
            >>> b.lol is b['lol']
            True
            >>> b.lol is getattr(b, 'lol')
            True
        """
        try:
            # Throws exception if not in prototype chain
            return object.__getattribute__(self, k)
        except AttributeError:
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)


    def __setattr__(self, k, v):
        """ Sets attribute k if it exists, otherwise sets key k. A KeyError
            raised by set-item (only likely if you subclass Munch) will
            propagate as an AttributeError instead.
            >>> b = AttrDict(foo='bar', this_is='useful when subclassing')
            >>> hasattr(b.values, '__call__')
            True
            >>> b.values = 'uh oh'
            >>> b.values
            'uh oh'
            >>> b['values']
            Traceback (most recent call last):
                ...
            KeyError: 'values'
        """
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                self[k] = v
            except:
                raise AttributeError(k)
        else:
            object.__setattr__(self, k, v)


    def __delattr__(self, k):
        """ Deletes attribute k if it exists, otherwise deletes key k. A KeyError
            raised by deleting the key--such as when the key is missing--will
            propagate as an AttributeError instead.
            >>> b = AttrDict(lol=42)
            >>> del b.lol
            >>> b.lol
            Traceback (most recent call last):
                ...
            AttributeError: lol
        """
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                del self[k]
            except KeyError:
                raise AttributeError(k)
        else:
            object.__delattr__(self, k)


    def toDict(self):
        """ Recursively converts a munch back into a dictionary.
            >>> b = AttrDict(foo=AttrDict(lol=True), hello=42, ponies='are pretty!')
            >>> sorted(b.toDict().items())
            [('foo', {'lol': True}), ('hello', 42), ('ponies', 'are pretty!')]
            See unmunchify for more info.
        """
        return unmunchify(self)


    @property
    def __dict__(self):
        return self.toDict()


    def __repr__(self):
        """ Invertible* string-form of a Munch.
            >>> b = AttrDict(foo=AttrDict(lol=True), hello=42, ponies='are pretty!')
            >>> print (repr(b))
            Munch({'ponies': 'are pretty!', 'foo': Munch({'lol': True}), 'hello': 42})
            >>> eval(repr(b))
            Munch({'ponies': 'are pretty!', 'foo': Munch({'lol': True}), 'hello': 42})
            >>> with_spaces = AttrDict({1: 2, 'a b': 9, 'c': AttrDict({'simple': 5})})
            >>> print (repr(with_spaces))
            Munch({'a b': 9, 1: 2, 'c': Munch({'simple': 5})})
            >>> eval(repr(with_spaces))
            Munch({'a b': 9, 1: 2, 'c': Munch({'simple': 5})})
            (*) Invertible so long as collection contents are each repr-invertible.
        """
        return '{0}({1})'.format(self.__class__.__name__, dict.__repr__(self))


    def __dir__(self):
        return list(iterkeys(self))


    def __getstate__(self):
        """ Implement a serializable interface used for pickling.
        See https://docs.python.org/3.6/library/pickle.html.
        """
        return {k: v for k, v in self.items()}


    def __setstate__(self, state):
        """ Implement a serializable interface used for pickling.
        See https://docs.python.org/3.6/library/pickle.html.
        """
        self.clear()
        self.update(state)


    __members__ = __dir__  # for python2.x compatibility


    @classmethod
    def fromDict(cls, d):
        """ Recursively transforms a dictionary into a Munch via copy.
            >>> b = AttrDict.fromDict({'urmom': {'sez': {'what': 'what'}}})
            >>> b.urmom.sez.what
            'what'
            See munchify for more info.
        """
        return munchify(d, cls)


    def copy(self):
        return type(self).fromDict(self)


class AutoAttrDict(AttrDict):
    def __setattr__(self, k, v):
        """ Works the same as Munch.__setattr__ but if you supply
            a dictionary as value it will convert it to another Munch.
        """
        if isinstance(v, dict) and not isinstance(v, (AutoAttrDict, AttrDict)):
            v = munchify(v, AutoAttrDict)
        super(AutoAttrDict, self).__setattr__(k, v)


class DefaultAttrDict(AttrDict):
    """
    A Munch that returns a user-specified value for missing keys.
    """


    def __init__(self, *args, **kwargs):
        """ Construct a new DefaultMunch. Like collections.defaultdict, the
            first argument is the default value; subsequent arguments are the
            same as those for dict.
        """
        # Mimic collections.defaultdict constructor
        if args:
            default = args[0]
            args = args[1:]
        else:
            default = None
        super(DefaultAttrDict, self).__init__(*args, **kwargs)
        self.__default__ = default


    def __getattr__(self, k):
        """ Gets key if it exists, otherwise returns the default value."""
        try:
            return super(DefaultAttrDict, self).__getattr__(k)
        except AttributeError:
            return self.__default__


    def __setattr__(self, k, v):
        if k == '__default__':
            object.__setattr__(self, k, v)
        else:
            return super(DefaultAttrDict, self).__setattr__(k, v)


    def __getitem__(self, k):
        """ Gets key if it exists, otherwise returns the default value."""
        try:
            return super(DefaultAttrDict, self).__getitem__(k)
        except KeyError:
            return self.__default__


    def __getstate__(self):
        """ Implement a serializable interface used for pickling.
        See https://docs.python.org/3.6/library/pickle.html.
        """
        return (self.__default__, {k: v for k, v in self.items()})


    def __setstate__(self, state):
        """ Implement a serializable interface used for pickling.
        See https://docs.python.org/3.6/library/pickle.html.
        """
        self.clear()
        default, state_dict = state
        self.update(state_dict)
        self.__default__ = default


    @classmethod
    def fromDict(cls, d, default=None):
        # pylint: disable=arguments-differ
        return munchify(d, factory=lambda d_: cls(default, d_))


    def copy(self):
        return type(self).fromDict(self, default=self.__default__)


    def __repr__(self):
        return '{0}({1!r}, {2})'.format(
            type(self).__name__, self.__undefined__, dict.__repr__(self))


class DefaultFactoryAttrDict(defaultdict, AttrDict):
    """ A Munch that calls a user-specified function to generate values for
        missing keys like collections.defaultdict.
        >>> b = DefaultFactoryAttrDict(list, {'hello': 'world!'})
        >>> b.hello
        'world!'
        >>> b.foo
        []
        >>> b.bar.append('hello')
        >>> b.bar
        ['hello']
    """


    def __init__(self, default_factory, *args, **kwargs):
        # pylint: disable=useless-super-delegation
        super(DefaultFactoryAttrDict, self).__init__(default_factory, *args, **kwargs)


    @classmethod
    def fromDict(cls, d, default_factory):
        # pylint: disable=arguments-differ
        return munchify(d, factory=lambda d_: cls(default_factory, d_))


    def copy(self):
        return type(self).fromDict(self, default_factory=self.default_factory)


    def __repr__(self):
        factory = self.default_factory.__name__
        return '{0}({1}, {2})'.format(
            type(self).__name__, factory, dict.__repr__(self))


# While we could convert abstract types like Mapping or Iterable, I think
# munchify is more likely to "do what you mean" if it is conservative about
# casting (ex: isinstance(str,Iterable) == True ).
#
# Should you disagree, it is not difficult to duplicate this function with
# more aggressive coercion to suit your own purposes.

def munchify(x, factory=AttrDict):
    """ Recursively transforms a dictionary into a Munch via copy.
        >>> b = munchify({'urmom': {'sez': {'what': 'what'}}})
        >>> b.urmom.sez.what
        'what'
        munchify can handle intermediary dicts, lists and tuples (as well as
        their subclasses), but ymmv on custom datatypes.
        >>> b = munchify({ 'lol': ('cats', {'hah':'i win again'}),
        ...         'hello': [{'french':'salut', 'german':'hallo'}] })
        >>> b.hello[0].french
        'salut'
        >>> b.lol[1].hah
        'i win again'
        nb. As dicts are not hashable, they cannot be nested in sets/frozensets.
    """
    if isinstance(x, dict):
        return factory((k, munchify(v, factory)) for k, v in iteritems(x))
    elif isinstance(x, (list, tuple)):
        return type(x)(munchify(v, factory) for v in x)
    else:
        return x


def unmunchify(x):
    """ Recursively converts a Munch into a dictionary.
        >>> b = AttrDict(foo=AttrDict(lol=True), hello=42, ponies='are pretty!')
        >>> sorted(unmunchify(b).items())
        [('foo', {'lol': True}), ('hello', 42), ('ponies', 'are pretty!')]
        unmunchify will handle intermediary dicts, lists and tuples (as well as
        their subclasses), but ymmv on custom datatypes.
        >>> b = AttrDict(foo=['bar', AttrDict(lol=True)], hello=42,
        ...         ponies=('are pretty!', AttrDict(lies='are trouble!')))
        >>> sorted(unmunchify(b).items()) #doctest: +NORMALIZE_WHITESPACE
        [('foo', ['bar', {'lol': True}]), ('hello', 42), ('ponies', ('are pretty!', {'lies': 'are trouble!'}))]
        nb. As dicts are not hashable, they cannot be nested in sets/frozensets.
    """
    if isinstance(x, dict):
        return dict((k, unmunchify(v)) for k, v in iteritems(x))
    elif isinstance(x, (list, tuple)):
        return type(x)(unmunchify(v) for v in x)
    else:
        return x


# Serialization

try:
    try:
        import json
    except ImportError:
        import simplejson as json


    def toJSON(self, **options):
        """ Serializes this Munch to JSON. Accepts the same keyword options as `json.dumps()`.
            >>> b = AttrDict(foo=AttrDict(lol=True), hello=42, ponies='are pretty!')
            >>> json.dumps(b) == b.toJSON()
            True
        """
        return json.dumps(self, **options)


    AttrDict.toJSON = toJSON

except ImportError:
    pass

# try:
#     # Attempt to register ourself with PyYAML as a representer
#     import yaml
#     from yaml.representer import Representer, SafeRepresenter
#
#
#     def from_yaml(loader, node):
#         """ PyYAML support for Munches using the tag `!munch` and `!munch.Munch`.
#             >>> import yaml
#             >>> yaml.load('''
#             ... Flow style: !munch.Munch { Clark: Evans, Brian: Ingerson, Oren: Ben-Kiki }
#             ... Block style: !munch
#             ...   Clark : Evans
#             ...   Brian : Ingerson
#             ...   Oren  : Ben-Kiki
#             ... ''') #doctest: +NORMALIZE_WHITESPACE
#             {'Flow style': Munch(Brian='Ingerson', Clark='Evans', Oren='Ben-Kiki'),
#              'Block style': Munch(Brian='Ingerson', Clark='Evans', Oren='Ben-Kiki')}
#             This module registers itself automatically to cover both Munch and any
#             subclasses. Should you want to customize the representation of a subclass,
#             simply register it with PyYAML yourself.
#         """
#         data = Munch()
#         yield data
#         value = loader.construct_mapping(node)
#         data.update(value)
#
#
#     def to_yaml_safe(dumper, data):
#         """ Converts Munch to a normal mapping node, making it appear as a
#             dict in the YAML output.
#             >>> b = Munch(foo=['bar', Munch(lol=True)], hello=42)
#             >>> import yaml
#             >>> yaml.safe_dump(b, default_flow_style=True)
#             '{foo: [bar, {lol: true}], hello: 42}\\n'
#         """
#         return dumper.represent_dict(data)
#
#
#     def to_yaml(dumper, data):
#         """ Converts Munch to a representation node.
#             >>> b = Munch(foo=['bar', Munch(lol=True)], hello=42)
#             >>> import yaml
#             >>> yaml.dump(b, default_flow_style=True)
#             '!munch.Munch {foo: [bar, !munch.Munch {lol: true}], hello: 42}\\n'
#         """
#         return dumper.represent_mapping(u('!munch.Munch'), data)
#
#
#     yaml.add_constructor(u('!munch'), from_yaml)
#     yaml.add_constructor(u('!munch.Munch'), from_yaml)
#
#     SafeRepresenter.add_representer(Munch, to_yaml_safe)
#     SafeRepresenter.add_multi_representer(Munch, to_yaml_safe)
#
#     Representer.add_representer(Munch, to_yaml)
#     Representer.add_multi_representer(Munch, to_yaml)
#
#
#     # Instance methods for YAML conversion
#     def toYAML(self, **options):
#         """ Serializes this Munch to YAML, using `yaml.safe_dump()` if
#             no `Dumper` is provided. See the PyYAML documentation for more info.
#             >>> b = Munch(foo=['bar', Munch(lol=True)], hello=42)
#             >>> import yaml
#             >>> yaml.safe_dump(b, default_flow_style=True)
#             '{foo: [bar, {lol: true}], hello: 42}\\n'
#             >>> b.toYAML(default_flow_style=True)
#             '{foo: [bar, {lol: true}], hello: 42}\\n'
#             >>> yaml.dump(b, default_flow_style=True)
#             '!munch.Munch {foo: [bar, !munch.Munch {lol: true}], hello: 42}\\n'
#             >>> b.toYAML(Dumper=yaml.Dumper, default_flow_style=True)
#             '!munch.Munch {foo: [bar, !munch.Munch {lol: true}], hello: 42}\\n'
#         """
#         opts = dict(indent=4, default_flow_style=False)
#         opts.update(options)
#         if 'Dumper' not in opts:
#             return yaml.safe_dump(self, **opts)
#         else:
#             return yaml.dump(self, **opts)
#
#
#     def fromYAML(*args, **kwargs):
#         return munchify(yaml.load(*args, **kwargs))
#
#
#     Munch.toYAML = toYAML
#     Munch.fromYAML = staticmethod(fromYAML)
#
# except ImportError:
#     pass



class JSONNumpyEncoder(json.JSONEncoder):
    """
    Encodes objects that have numpy arrays into json objects.

    The json_numpy_object_hook() function can be used to load the dumped objects.


    Usage:

        import json

        dumped = json.dumps(object, cls=JSONNumpyEncoder)
        loaded = json.loads(dumped, object_hook=json_numpy_object_hook)
    """

    def default(self, obj):
        """
        if input object is a ndarray it will be converted into a dict holding dtype, shape and the data base64 encoded
        """
        if isinstance(obj, np.ndarray):
            data_b64 = base64.b64encode(np.ascontiguousarray(obj).data)
            return dict(__ndarray__=data_b64.decode('ascii'),
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError

        if isinstance(obj, np.int64):
            return int(obj)

        if isinstance(obj, np.float64):
            return float(obj)

        return super(JSONNumpyEncoder, self).default(obj)


def json_numpy_object_hook(dct):
    """
    Decodes a previously encoded numpy ndarray with proper shape and dtype

    Usage:

        import json

        dumped = json.dumps(object, cls=JSONNumpyEncoder)
        loaded = json.loads(dumped, object_hook=json_numpy_object_hook)

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct


def set_dict_default_values(*args, is_copy=True):

    args = list(args)

    for arg_idx in range(len(args)-1, 0, -1):

        if args[arg_idx-1] is None:
            args[arg_idx-1] = {}

        if is_copy:
            args[arg_idx-1] = args[arg_idx-1].copy()
        else:
            args[arg_idx-1] = args[arg_idx-1]

        for def_key, def_item in args[arg_idx].items():

            if not def_key in args[arg_idx-1]:
                # add default item if not found target
                args[arg_idx-1][def_key] = def_item
            elif isinstance(def_item, collections.Mapping) and isinstance(args[arg_idx][def_key], collections.Mapping):
                # if the value is a dictionary in the default and the target, then also set default values for it
                args[arg_idx-1][def_key] = set_dict_default_values(args[arg_idx-1][def_key], def_item, is_copy=True)

    return args[0]
