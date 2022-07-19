#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import hoomd
import hoomd.md
import hoomd.dlext

import pysages

from pysages.colvars import Component
from pysages.methods import Unbias, HistogramLogger


def generate_context(**kwargs):
    hoomd.context.initialize()
    context = hoomd.context.SimulationContext()
    with context:
        hoomd.init.read_gsd("start.gsd")
        hoomd.md.integrate.nve(group=hoomd.group.all())
        hoomd.md.integrate.mode_standard(dt=0.01)

        nl = hoomd.md.nlist.cell()
        dpd = hoomd.md.pair.dpd(r_cut=1, nlist=nl, seed=42, kT=1.0)
        dpd.pair_coeff.set("A", "A", A=kwargs.get("A", 5.0), gamma=kwargs.get("gamma", 1.0))
    return context


def main():
    cvs = [Component([0], 2)]
    cvs += [Component([0], 1)]
    cvs += [Component([0], 0)]

    center_cv = [0.0]
    center_cv += [1.0, -0.3]

    method = Unbias(cvs)
    callback = HistogramLogger(100)

    raw_result = pysages.run(method, generate_context, int(1e3), callback)

    print(np.asarray(raw_result.callbacks[0].data))


if __name__ == "__main__":
    main()
