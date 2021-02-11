"""Datatype validation."""

__all__ = ['failures', 'is_valid']

from collections import defaultdict

from datatype.tools import walk


def is_valid(datatype, value):
    """Return boolean representing validity of `value` against `datatype`."""
    return not failures(datatype, value)


def failures(datatype, value):
    """Return list of failures (if any) validating `value` against `datatype`.

    Params:
        `datatype`: Datatype to validate value against.  See README.markdown
            for examples.
        `value`: Value to validate
        `path`: Used internally for location of failures.

    Example:
        >>> failures('int', 'foo')
        ['expected int, got str']
    """
    fails = []

    def validate(path, *args):
        msg = '%s: %%s' % path if path else '%s'
        fails.extend(msg % m for m in validate_step(*args) or [])

    walk(datatype, value, validate)

    return fails


def validate_step(datatype, value, options):
    """Validate simple value in datatype."""
    dtype, vtype = type(datatype), type(value)

    # nulls
    if vtype is type(None):
        if 'nullable' not in options:
            return ['unexpected null for non-nullable type']

    # primitives
    elif dtype == str:
        if not isinstance(value, primitives[datatype]):
            return ['expected %s, got %s' % (datatype, vtype.__name__)]

    # lists & tuples
    elif dtype == list:
        dlen, vlen = len(datatype), len(value)
        if dlen > 1 and dlen != vlen:
            error = 'missing required' if dlen > vlen else 'unexpected'
            return ['%s value at index %s' % (error, i)
                    for i in xrange(min(dlen, vlen), max(dlen, vlen))]

    # objects (dictionaries)
    elif dtype in (defaultdict, dict):
        if vtype not in (defaultdict, dict):
            return ['expected dict, got %s' % vtype.__name__]

        optional = lambda x: x.startswith('optional ')
        all_keys = (k.replace('optional ', '', 1) if optional(k) else k
               for k in datatype)
        required_keys = (k for k in datatype
               if not (optional(k) or k == '_any_'))

        failures = ['missing required property: "%s"' % k
                    for k in set(required_keys) - set(value)]

        if '_any_' not in datatype:
            failures += ['unexpected property "%s"' % k
                         for k in set(value) - set(all_keys)]

        return failures


primitives = {
        'int':   (int,),
        'float': (float,),
        'str':   (str, unicode),
        'bool':  (bool,)
    }

