"""Docblock manipulation utilities."""

from pprint import pformat


def append_to_docs(fn, text):
    """Append text to a functions existing docblock."""
    if not text:
        return
    if fn.__doc__:
        min_indent = _getindent(fn.__doc__)
        fn.__doc__ = '%s\n\n%s' % (fn.__doc__, _indent(text, min_indent))
    else:
        fn.__doc__ = text


def append_var_to_docs(fn, label, value):
    """Append text & pformatted value to docblock."""
    value_width = 76 - _getindent(fn.__doc__)
    append_to_docs(
            fn,
            "%s:\n%s" % (
                label,
                _indent(pformat(value, width=value_width))
            )
        )


def include_docs_from(source_function):
    """Decorator copying documentation from one function onto another."""
    def decorator(dest_function):
        append_to_docs(dest_function, source_function.__doc__)
        return dest_function
    return decorator


def _indent(string, indent_level=4):
    """Indent each line by `indent_level` of spaces."""
    return '\n'.join('%s%s' % (' '*indent_level, x) for x in
            string.splitlines())


def _getindent(string):
    try:
        lines = string.splitlines()

        # drop first line if it has no indent level
        if _nspaces(lines[0]) == 0:
            lines.pop(0)

        indent_levels = (_nspaces(x) for x in lines if x)
        return min(indent_levels) or 0
    except (AttributeError, ValueError):
        # Things that don't look like strings and strings with no
        # indentation should report indentation of 0
        return 0


def _nspaces(line):
    for idx, char in enumerate(line):
        if char != ' ':
            return idx

