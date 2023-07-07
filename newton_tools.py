#
# The actual computing is happening here
#
# Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

from collections.abc import Callable

import numpy as np
from pydantic import validate_arguments

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def newton_method(
        func: Callable[[np.ndarray], np.ndarray],
        func_prime: Callable[[np.ndarray], np.ndarray],
        z0: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-6
        ) -> np.ndarray:
    """
    Perform Newton Iteration on `func`

    Parameters
    ----------
    partition : str
        Name of SLURM partition/queue to start workers in. Use the command `sinfo`
        in the terminal to see a list of available SLURM partitions on the ESI HPC
        cluster.
    """

    z = z0
    frac = np.ones(z.shape)
    for i in range(max_iter):
        dz = func(z) / func_prime(z)
        frac[np.abs[dz] < tol] = i / max_iter
        z -= dz
    return frac

def plot_fractal(
        frac: np.ndarray,
        cmap: str = "viridis",
        interp: str = "bilinear"
        ) -> None:

    pass

def create_grid(
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        res_x: int = 256,
        res_y: int = 256,
        ) -> np.ndarray:


    x = np.linspace(xmin, xmax, res_x)
    y = np.linspace(ymin, ymax, res_y)
    X,Y = np.meshgrid(x, y)
    Z = X + Y*1j
    return Z
