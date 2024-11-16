from typing import Callable
import warnings
from collections import defaultdict
from contextlib import contextmanager
from abc import abstractmethod

from autograd.util import subvals, toposort
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class Node:
    r"""An autograd node."""
    __slots__ = []

    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        pass

    @abstractmethod
    def initialize_root(self, *args, **kwargs):
        pass

    @classmethod
    def new_root(cls, *args, **kwargs):
        root = cls.__new__(cls)
        root.initialize_root(*args, **kwargs)
        return root

def trace(start_node, fun, x):
    r"""Traces the execution of function `f` with boxed argument `x`.

    Note that the boxed argument may be a `SequenceBox` or `DictBox`, in which
    case it is a container of multiple arguments."""
    with trace_stack.new_trace() as t:
        logger.info("New trace (stack_id=%s) started for function %s at %s", t, fun, x)
        start_box = new_box(x, t, start_node)
        end_box = fun(start_box)
        if isbox(end_box) and end_box._trace == start_box._trace:
            return end_box._value, end_box._node
        else:
            warnings.warn("Output seems independent of input.")
            return end_box, None



def primitive(f_raw: Callable):
    """Wraps a function so that its gradient can be specified and its invocation
    can be recorded.

    If the function has no boxed arguments, it is treated as a primitive. If
    the function has boxed arguments (and it is not part of `notrace_primitives`),
    the function is executed with boxes unwrapped, and the output is boxed."""

    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        # Fetch the (boxed) arguments corresponding to the most recent trace:
        boxed_args, trace, node_constructor = find_top_boxed_args(args)

        if boxed_args:
            # Unwrap boxed arguments corresponding to most recent trace:
            # TODO(manan): it's still unclear how multiple traces work...
            argvals = subvals(args, [(argnum, box._value)
                                     for argnum, box in boxed_args])

            if f_wrapped in notrace_primitives[node_constructor]:
                # If we do not trace this primitive, return directly:
                return f_wrapped(*argvals, **kwargs)

            # Otherwise, compute the output and box it (with the same
            # node type as the input arguments):
            ans = f_wrapped(*argvals, **kwargs)

            parents = tuple(box._node for _, box in boxed_args)
            argnums = tuple(argnum for argnum, _ in boxed_args)
            node = node_constructor(ans, f_wrapped, argvals, kwargs, argnums,
                                    parents)
            return new_box(ans, trace, node)
        else:
            # If no boxed arguments, just return the function, we do not need
            # to differentiate wrt any arguments here:
            return f_raw(*args, **kwargs)

    # Metadata:
    f_wrapped.fun = f_raw
    f_wrapped._is_autograd_primitive = True
    return f_wrapped


notrace_primitives = defaultdict(set)


def register_notrace(trace_type, primitive_fun):
    notrace_primitives[trace_type].add(primitive_fun)


def notrace_primitive(f_raw):

    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        argvals = map(getval, args)
        return f_raw(*argvals, **kwargs)

    f_wrapped._is_primitive = True
    return f_wrapped


def find_top_boxed_args(args: list):
    r"""Find the topmost boxed argument in the list of arguments; that is,
    for a function that takes a list of arguments that has been traced
    multiple times, find the arguments that correspond to the most recent
    trace."""

    # Traces are ascending: 0 is the lowest, N is the highest (for N
    # traces):
    top_trace = -1
    top_boxes = []
    top_node_type = None

    # Find all (boxed) arguments corresponding to the most recent
    # trace:
    for argnum, arg in enumerate(args):
        if isbox(arg):  # fast-path isinstance check...
            trace = arg._trace
            if trace > top_trace:
                top_boxes = [(argnum, arg)]
                top_trace = trace
                top_node_type = type(arg._node)
            elif trace == top_trace:
                top_boxes.append((argnum, arg))

    # Return (boxed) arguments, their top trace, and the type of
    # autograd node (e.g. VJPNode):
    return top_boxes, top_trace, top_node_type


class TraceStack:
    r"""Manages a stack of traces (tracing traces, for example). This is done
    simply, by maintaining a `top` value that is incremented with each trace
    start."""

    def __init__(self):
        self.top = -1

    @contextmanager
    def new_trace(self):
        self.top += 1
        yield self.top
        self.top -= 1


trace_stack = TraceStack()


class Box:
    r"""A box wraps an argument as part of an autograd execution trace."""
    type_mappings = {}
    types = set()

    __slots__ = ["_value", "_trace", "_node"]

    def __init__(self, value, trace, node):
        self._value = value
        self._node = node
        self._trace = trace

    def __bool__(self):
        return bool(self._value)

    __nonzero__ = __bool__

    def __str__(self):
        return f"{type(self).__name__}(val={str(self._value)})"

    @classmethod
    def register(cls, value_type):
        Box.types.add(cls)
        Box.type_mappings[value_type] = cls
        Box.type_mappings[cls] = cls


box_type_mappings = Box.type_mappings


def new_box(value, trace, node):
    try:
        out = box_type_mappings[type(value)](value, trace, node)
        logger.info("Created box %s", out)
        return out
    except KeyError:
        raise TypeError(f"Can't differentiate w.r.t. type {type(value)}")


box_types = Box.types
isbox = lambda x: type(
    x) in box_types  # almost 3X faster than isinstance(x, Box)
getval = lambda x: getval(x._value) if isbox(x) else x
