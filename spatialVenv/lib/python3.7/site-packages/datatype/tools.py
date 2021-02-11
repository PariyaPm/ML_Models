"""Datatype utility functions.  Used by datatype validation and coercion."""

from collections import defaultdict
from functools import partial
from itertools import count


class NewValue(object):
    """Returned from walk-callback when we want to replace the value."""

    __slots__ = ('value',)
    
    def __init__(self, value):
        self.value = value


def dict_datatypes(datatype):
    default = datatype.get('_any_')
    datatypes = defaultdict(lambda: default)

    for key, val in datatype.iteritems():
        if isinstance(val, str):
            key, _ = parse_dict_key(key)
        else:
            key = key
        datatypes[key] = val

    return datatypes


def walk(datatype, value, callback, path='', options=None):
    """Walk the value and datatype with the given callback.

    Example callback: lambda path, datatype, actual, options: None"""
    options = options or []
    if isinstance(datatype, str):
        datatype, parsed_options = parse_primitive(datatype)
        options += parsed_options

    new_value = callback(path, datatype, value, options)

    # Are we replacing the value?
    if isinstance(new_value, NewValue):
        value = new_value.value

    # Walk lists and tuples
    if are_type(list, datatype, value):
        dt_len = len(datatype)
        mk_path = lambda i: joinpaths(path, '[%d]' % i)

        if dt_len == 1:   # list of `a`
            value = [walk(datatype[0], v, callback, mk_path(i))
                     for i, v in enumerate(value)]
        elif dt_len > 1:  # tuple
            value = [walk(d, v, callback, mk_path(i))
                     for i, d, v in zip(count(), datatype, value)]

    # Walk objects (dictionaries)
    elif are_type(dict, datatype, value):
        key_dts = dict_datatypes(datatype)
        mk_path = lambda k: joinpaths(path, k, '.')

        value = dict((k, walk(key_dts[k], v, callback, mk_path(k)))
            for k, v in value.iteritems())

    return value


def are_type(type_, *vars_):
    return all(isinstance(v, type_) for v in vars_)


def joinpaths(p1, p2, delim=None):
    return '%s%s%s' % (p1, delim, p2) if delim and p1 else '%s%s' % (p1, p2)


def parse_name_options(key, possible_options):
    """Pull dictionary key options from key and return both as tuple.

    Example:
        >>> parse_name_options("optional foo", ['optional'])
        ('foo', ['optional'])
    """
    key_words = iter(key.split(' '))
    options = []
    for word in key_words:
        if word in possible_options and word not in options:
            options.append(word)
        else:
            return (' '.join([word] + list(key_words)), options)


parse_dict_key = partial(parse_name_options, possible_options=['optional'])
parse_primitive = partial(parse_name_options, possible_options=['nullable'])

