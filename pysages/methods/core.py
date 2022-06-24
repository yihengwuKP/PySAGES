# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from abc import ABC, abstractmethod
from functools import reduce
from inspect import getfullargspec
from operator import or_
from typing import Callable, Optional, Union

from jax import grad, jit
from plum import parametric

from pysages.backends import ContextWrapper
from pysages.grids import Grid, NoGrid, build_grid, get_info
from pysages.colvars.core import build
from pysages.utils import dispatch, identity


#  Base Classes
#  ============


class SamplingMethod(ABC):
    """
    Abstract base class for all sampling methods.

    Defines the constructor that expects the collective variables,
    the build method to initialize the GPU execution for the biasing,
    and the run method that executes the simulation run.
    All these are intended to be enhanced/overwritten by inheriting classes.
    """

    __special_args__ = set()
    snapshot_flags = set()

    def __init__(self, cvs, **kwargs):
        self.cvs = cvs
        self.cv = build(*cvs, grad=kwargs.get("cv_grad", grad))
        self.requires_box_unwrapping = reduce(
            or_, (cv.requires_box_unwrapping for cv in cvs), False
        )
        self.kwargs = kwargs

    def __getstate__(self):
        return default_getstate(self)

    def __setstate__(self, state):
        default_setstate(self, state)

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        """
        Returns the snapshot, and two functions: `initialize` and `update`.
        `initialize` is intended to allocate any run time information required
        by `update`, while `update` is intended to be called after each call to
        the wrapped context's `run` method.
        """
        pass


class GriddedSamplingMethod(SamplingMethod):
    __special_args__ = {"grid"}

    def __init__(self, cvs, grid, **kwargs):
        check_dims(cvs, grid)
        super().__init__(cvs, **kwargs)
        self.grid = grid

    def __getstate__(self):
        return (get_info(self.grid), *default_getstate(self))

    def __setstate__(self, state):
        grid_args, args, kwargs = state
        args["grid"] = build_grid(*grid_args)
        default_setstate(self, (args, kwargs))

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


class NNSamplingMethod(GriddedSamplingMethod):
    def __init__(self, cvs, grid, topology, **kwargs):
        super().__init__(cvs, grid, **kwargs)
        self.topology = topology

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


@parametric
class Result:
    @classmethod
    def __infer_type_parameter__(self, method, *args):
        return type(method)

    @dispatch
    def __init__(self, method: SamplingMethod, states, callbacks=None):
        self.method = method
        self.states = states
        self.callbacks = callbacks


#  Main functions
#  ==============


@dispatch
def run(
    method: SamplingMethod,
    context_generator: Callable,
    timesteps: Union[int, float],
    callback: Optional[Callable] = None,
    context_args: Optional[dict] = None,
    post_run_action: Optional[Callable] = None,
    **kwargs
):
    """
    Base implementation for running a single simulation with the specified `SamplingMethod`.

    Arguments
    ---------
    method: SamplingMethod

    context_generator: Callable
        User defined function that sets up a simulation context with the backend.
        Must return an instance of `hoomd.context.SimulationContext` for HOOMD-blue
        and `openmm.Simulation` for OpenMM. The function gets `context_args`
        unpacked for additional user arguments.

    timesteps: int
        Number of time steps the simulation is running.

    callback: Optional[Callable] = None
        Allows for user defined actions into the simulation workflow of the method.
        `kwargs` gets passed to the backend `run` function.

    context_args: Optional[dict] = None
        Arguments to pass down to `context_generator` to setup the simulation context.

    post_run_action: Optional[Callable] = None
        Callable function that enables actions after the run execution of PySAGES.
        Actions are executed inside the generated context. Example uses for this
        include writing a final configuration file. This function gets `context_args`
        unpacked just like `context_generator`.
    """
    timesteps = int(timesteps)
    context_args = {} if context_args is None else context_args

    context = context_generator(**context_args)
    wrapped_context = ContextWrapper(context, method, callback)

    with wrapped_context:
        wrapped_context.run(timesteps, **kwargs)
        if post_run_action:
            post_run_action(**context_args)

    state = wrapped_context.sampler.state
    return Result(method, state, callback)


@dispatch.abstract
def analyze(result: Result):
    pass


#  Utils
#  =====


def default_getstate(method: SamplingMethod):
    init_args = set(getfullargspec(method.__init__).args[1:]) - method.__special_args__
    return {key: method.__dict__[key] for key in init_args}, method.kwargs


def default_setstate(method, state):
    args, kwargs = state
    method.__init__(**args, **kwargs)


@dispatch
def check_dims(cvs, grid: Grid):
    if len(cvs) != grid.shape.size:
        raise ValueError("Grid and Collective Variable dimensions must match.")


@dispatch
def check_dims(cvs, grid: NoGrid):
    pass


def generalize(concrete_update, helpers, jit_compile=True):
    if jit_compile:
        _jit = jit
    else:
        _jit = identity

    _update = _jit(concrete_update)

    def update(snapshot, state):
        return _update(state, helpers.query(snapshot))

    return _jit(update)
