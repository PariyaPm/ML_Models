import logging

from functools import wraps

from doctools import append_var_to_docs

from datatype.validation import failures


logger = logging.getLogger('datatype')

class BadReturnValueError(Exception):
    """Raised when `returns` decorator encounters a return value
    not matching it's given datatype."""

    def __init__(self, fails=None):
        # List of things that went wrong in validation
        self.failures = fails or []


def returns(dfn, strict=True):
    """Make decorators to watch return values of functions to ensure
    they match the given datatype definition.

    Optional Arguments:
        strict: if false, unexpected values on dictionaries will not raise an
        exception

    Example:
        >>> @returns('int')
        ... def myfunction():
        ...     return "bad return value"
        >>> myfunction()
        Traceback (most recent call last):
        BadReturnValueError
    """
    def decorator(fn):

        # Add return-datatype info to function doc-block
        append_var_to_docs(fn, "Return datatype", dfn)

        @wraps(fn)
        def wrapped_function(*args, **kwargs):
            ret = fn(*args, **kwargs)

            # Check for failure and raise
            fails = failures(dfn, ret)
            if fails:
                if strict or _get_bad_values(fails):
                    raise BadReturnValueError(fails)
                else:
                    logger.warning(fails)

            # All is well, return as usual
            return ret
        return wrapped_function
    return decorator


def returns_iter(dfn, strict=True):
    """Validate output of iterator/generator function.

    Note: exceptions for bad return datatypes will not be raised until the bad
    value of the iterator is consumed.

    Optional Arguments:
        strict: if false, unexpected values on dictionaries will not raise an
        exception

    Example:
        >>> @returns_iter('str')
        ... def myfunction():
        ...     for x in range(3):
        ...         yield "number %s" % x
        >>> list(myfunction())  # no error
        ['number 0', 'number 1', 'number 2']
    """
    def decorator(fn):
        append_var_to_docs(fn, "Return datatype (iterator of)", dfn)

        @wraps(fn)
        def wrapped_function(*args, **kwargs):
            for value in fn(*args, **kwargs):
                fails = failures(dfn, value)
                if fails:
                    if strict or _get_bad_values(fails):
                        raise BadReturnValueError(fails)
                    else:
                        logger.warning(fails)

                yield value
        return wrapped_function
    return decorator


def _get_bad_values(fails):
    return [x for x in fails if 'unexpected property' not in x]

