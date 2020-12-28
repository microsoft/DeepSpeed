import sys as _sys

from typing import List
from collections import _iskeyword  # type: ignore
from tensorboardX import SummaryWriter
import os

SUMMARY_WRITER_DIR_NAME = 'runs'


def get_sample_writer(name, base=".."):
    """Returns a tensorboard summary writer
    """
    return SummaryWriter(
        log_dir=os.path.join(base, SUMMARY_WRITER_DIR_NAME, name))


class TorchTuple(tuple):
    def to(self, device, non_blocking=False):
        raise NotImplementedError("")


_class_template = """\
from builtins import property as _property, tuple as _tuple
from operator import itemgetter as _itemgetter
from collections import OrderedDict

from turing.utils import TorchTuple

import torch

class {typename}(TorchTuple):
    '{typename}({arg_list})'

    __slots__ = ()

    _fields = {field_names!r}

    def __new__(_cls, {arg_list}):
        'Create new instance of {typename}({arg_list})'
        return _tuple.__new__(_cls, ({arg_list}))

    @classmethod
    def _make(cls, iterable, new=tuple.__new__, len=len):
        'Make a new {typename} object from a sequence or iterable'
        result = new(cls, iterable)
        if len(result) != {num_fields:d}:
            raise TypeError('Expected {num_fields:d} arguments, got %d' % len(result))
        return result

    def _replace(_self, **kwds):
        'Return a new {typename} object replacing specified fields with new values'
        result = _self._make(map(kwds.pop, {field_names!r}, _self))
        if kwds:
            raise ValueError('Got unexpected field names: %r' % list(kwds))
        return result

    def __repr__(self):
        'Return a nicely formatted representation string'
        return self.__class__.__name__ + '({repr_fmt})' % self

    @property
    def __dict__(self):
        'A new OrderedDict mapping field names to their values'
        return OrderedDict(zip(self._fields, self))

    def _asdict(self):
        '''Return a new OrderedDict which maps field names to their values.
           This method is obsolete.  Use vars(nt) or nt.__dict__ instead.
        '''
        return self.__dict__

    def __getnewargs__(self):
        'Return self as a plain tuple.  Used by copy and pickle.'
        return tuple(self)

    def __getstate__(self):
        'Exclude the OrderedDict from pickling'
        return None

    def to(self, device, non_blocking=False):
        _dict = self.__dict__.copy()
        new_dict = dict()
        for key, value in _dict.items():
            if isinstance(value, torch.Tensor):
                if device.type != 'cpu' and non_blocking and torch.cuda.is_available():
                    new_dict[key] = value.cuda(device, non_blocking=non_blocking)
                else:
                    new_dict[key] = value.to(device)
            else:
                new_dict[key] = value
        return {typename}(**new_dict)
{field_defs}
"""

_repr_template = '{name}=%r'

_field_template = '''\
    {name} = _property(_itemgetter({index:d}), doc='Alias for field number {index:d}')
'''


def namedtorchbatch(typename: str,
                    field_names: List[str],
                    verbose: bool = False,
                    rename: bool = False):
    """Returns a new subclass of tuple with named fields leveraging use of torch tensors.
    """

    # Validate the field names.  At the user's option, either generate an error
    # message or automatically replace the field name with a valid name.
    if isinstance(field_names, str):
        field_names = field_names.replace(',', ' ').split()
    field_names = list(map(str, field_names))
    if rename:
        seen: set = set()
        for index, name in enumerate(field_names):
            if (not name.isidentifier() or _iskeyword(name)
                    or name.startswith('_') or name in seen):
                field_names[index] = '_%d' % index
            seen.add(name)
    for name in [typename] + field_names:
        if not name.isidentifier():
            raise ValueError('Type names and field names must be valid '
                             'identifiers: %r' % name)
        if _iskeyword(name):
            raise ValueError('Type names and field names cannot be a '
                             'keyword: %r' % name)
    seen = set()
    for name in field_names:
        if name.startswith('_') and not rename:
            raise ValueError('Field names cannot start with an underscore: '
                             '%r' % name)
        if name in seen:
            raise ValueError('Encountered duplicate field name: %r' % name)
        seen.add(name)

    # Fill-in the class template
    class_definition = _class_template.format(
        typename=typename,
        field_names=tuple(field_names),
        num_fields=len(field_names),
        arg_list=repr(tuple(field_names)).replace("'", "")[1:-1],
        repr_fmt=', '.join(
            _repr_template.format(name=name) for name in field_names),
        field_defs='\n'.join(
            _field_template.format(index=index, name=name)
            for index, name in enumerate(field_names)))

    # Execute the template string in a temporary namespace and support
    # tracing utilities by setting a value for frame.f_globals['__name__']
    namespace = dict(__name__='namedtuple_%s' % typename)
    exec(class_definition, namespace)
    result = namespace[typename]
    result._source = class_definition  # type: ignore
    if verbose:
        print(result._source)  # type: ignore

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in enviroments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython).
    try:
        result.__module__ = _sys._getframe(1).f_globals.get(
            '__name__', '__main__')
    except (AttributeError, ValueError):
        pass

    return result
