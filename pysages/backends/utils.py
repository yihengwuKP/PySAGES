# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES


import ctypes
import numba

from numpy.ctypeslib import as_ctypes_type


def view(device_array):
    """Return a writable view of a JAX DeviceArray."""
    ptype = ctypes.POINTER(as_ctypes_type(device_array.dtype))
    addr = device_array.device_buffer.unsafe_buffer_pointer()
    ptr = ctypes.cast(ctypes.c_void_p(addr), ptype)
    return numba.carray(ptr, device_array.shape)
