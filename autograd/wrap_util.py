from autograd.util import subvals
from functools import wraps as fct_wraps

from typing import Any, Callable, Iterable, Union


def unary_to_nary(unary_operator: Callable):
    r"""There is a lot of wrapping going on here. Let's break it down:
        
    The input is `unary_operator`, a method that takes as arguments:
        * A function `fun`, taking an arbitrary number of arguments
        * An argument `x`, representing the arguments passed to function `fun`

    The output is an n-ary operator, which accepts a restriction of the arguments
    to `fun` (and fixes the others).
        * The caller sees a different signature, allowing specification of
            `argnum`.
        * The wrapped function sees two differences: `fun` is now `unary_f`
            accepting a restricted set of arguments (specifically,
            `len(argnum)` number of arguments, with the others fixed), and `x`
            is a sliced subset of the caller's arguments (based on `argnum`).
    """

    @fct_wraps(unary_operator)
    def nary_operator(
        fun: Callable,
        argnum: Union[int, tuple, list] = 0,
        *nary_op_args,
        **nary_op_kwargs,
    ):
        r"""Wraps `unary_operator`. The new method accepts the same function
        that `unary_operator` did, along with a slice of arguments that can be
        specified to slice inputs of `unary_operator`. 
        """

        # Wrap with `wrap_nary_f` (instead of `wraps`) so we can inject a custom
        # docstring:
        @wrap_nary_f(fun, unary_operator, argnum)
        def nary_f(*args, **kwargs):
            r"""Wraps `fun`, adjusting the arguments to `fun` to align with the
            argument numbers passed to `nary_operator`.

            Concretely, `argnum` corresponds to the arguments that the gradient
            of `fun` will be evaluated at; these will be represented as Boxes
            to track gradients. The other (free) arguments are not wrapped as
            Boxes."""

            @fct_wraps(fun)
            def unary_f(x):
                r"""Overwrites the locations specified by `argnum` in `fun`'s
                arguments to correspond to the values in `x`."""
                if isinstance(argnum, int):
                    # args[argnum] = x
                    subargs = subvals(args, [(argnum, x)])
                else:
                    assert isinstance(argnum, Iterable)
                    # for a_i in argnums: args[a_i] = x_i
                    subargs = subvals(args, zip(argnum, x))

                # Return an evaluation of `fun` with the overwritten
                # arguments:
                return fun(*subargs, **kwargs)

            if isinstance(argnum, int):
                x = args[argnum]
            else:
                x = tuple(args[i] for i in argnum)

            # Return an instance of `unary_operator` (the function we are wrapping),
            # with `x` being the sliced set of arguments that we select with
            # `argnum`. `fun` is wrapped with `unary_f`, which maintains references
            # to the arguments passed to `unary_operator` (so we can call `fun`)
            # and overrides the relevant parameters (those specified by `argnum`)
            # with whatever is passed in to the function. All other parameters are
            # fixed.
            return unary_operator(unary_f, x, *nary_op_args, **nary_op_kwargs)

        return nary_f

    return nary_operator


def wrap_nary_f(fun: Callable, op: Any, argnum: Union[int, tuple, list]):
    r"""Wrap an n-ary function with an interpretable docstring."""
    namestr = "{op}_of_{fun}_wrt_argnum_{argnum}"
    docstr = """\
    {op} of function {fun} with respect to argument number {argnum}. Takes the
    same arguments as {fun} but returns the {op}.
    """

    def _wraps(f):
        try:
            f.__name__ = namestr.format(fun=get_name(fun),
                                        op=op,
                                        argnum=argnum)
            f.__doc__ = docstr.format(
                fun=get_name(fun),
                doc=get_doc(fun),
                op=op,
                argnum=argnum,
            )
        finally:
            return f

    return _wraps


# Fetch the name and documentation of an arbitrary function `f`:
get_name = lambda f: getattr(f, "__name__", "[unknown name]")
get_doc = lambda f: getattr(f, "__doc__", "[no documentation]")
