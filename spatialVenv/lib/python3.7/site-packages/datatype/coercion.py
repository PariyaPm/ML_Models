"""Datatype coercion.

Provides:
    `coerce_value`: Returns best attempt at coercing value to given type.
"""

__all__ = ['coerce_value']

from datatype.tools import NewValue, walk
from datatype.validation import primitives


coercable_types = reduce(list.__add__,
        (list(x) for x in primitives.itervalues()))

coerce_to = dict((k, v[0]) for k, v in primitives.iteritems())


def coerce_value(datatype, value):
    """Attempt to coerce value to requested datatype.

    Example:
        >>> coerce_value("int", "5")
        5

    If coercion is not possible, the incoercible portion is left changed.

    Example:
        >>> coerce_value(['int'], ['1', '2', 'c'])
        [1, 2, 'c']
    """
    return walk(datatype, value, coerce_step)


def coerce_step(_, datatype, value, __):
    if type(datatype) == str and type(value) in coercable_types:
        try:
            return NewValue(coerce_to[datatype](value))
        except (TypeError, ValueError):
            # Because a errors in constructing these types is expected,
            # pass silently
            pass

