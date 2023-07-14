#
# Actual computational backbone for rendering Newton fractals
#
# Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

import os
from collections.abc import Callable
from colorsys import hls_to_rgb

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def newton_method(
        func: Callable[[np.ndarray], np.ndarray],
        func_prime: Callable[[np.ndarray], np.ndarray],
        z0: np.ndarray,
        max_iter: int = 10000,
        tol: float = 1e-6
        ) -> np.ndarray:
    """
    Perform Newton Iteration on `func`

    Parameters
    ----------
    func : callable
        Function to find roots of. Must be of the form ``y = f(x)``, where
        `x` and `y` are 1D/2D NumPy arrays.
    func_prime : callable
        Derivative of `func`. Must be of the same form as `func`: ``y = f'(x)``
        with 1D/2D NumPy arrays `x` and `y`.
    z0 : 1D/2D np.ndarray
        Initial value(s) for Newton iteration.
    max_iter: int
        Maximal number of Newton iterations to perform
    tol : float
        Convergence tolerance for stopping iteration

    Returns
    -------
    frac : 1D/2D np.ndarray
        Iteration number `k` normalized by `max_iter`, i.e.,
        ``frac = 1 - k / max_iter`` (higher is "better"). Thus, if
        Newton's method converged (result below tolerance `tol`)
        within `k` iterations given the starting point `z0`, then
        ``0 < frac = 1 - k / max_iter <= 1``, otherwise ``frac = 0``.
    """

    z = np.array(z0)
    frac = np.ones(z.shape)
    mask = np.ma.make_mask(frac)
    tol2 = 1e-9
    for i in range(max_iter):
        dfz = func_prime(z)
        dfz[np.abs(dfz) < tol2] = tol2
        dz = func(z) / dfz
        tmp = np.abs(dz) < tol
        frac[tmp * mask] = i / max_iter
        mask[tmp] = False
        if not np.any(mask):
            break
        z -= dz
    return 1 - frac


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def create_grid(
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        res_x: int = 256,
        res_y: int = 256,
        ) -> np.ndarray:
    """
    Create grid of complex numbers ``Z[m, n] = x[m] + i y[n]``

    Parameters
    ----------
    xmin : float
        Minimal value of real part ``Re(Z[m, n])``
    xmax : float
        Maximal value of real part ``Re(Z[m, n])``
    ymin : float
        Minimal value of imaginary part ``Im(Z[m, n])``
    ymax : float
        Maximal value of imaginary part ``Im(Z[m, n])``
    res_x : int
        Grid resolution on real axis, i.e., ``m = 1,...,res_x``
    res_y : int
        Grid resolution on imaginary axis, i.e., ``n = 1,...,res_y``

    Returns
    -------
    Z : 2D np.ndarray
        Grid of complex numbers ``Z[m, n] = x[m] + i y[n]`` with
        ``m = 1,...,res_x`` and ``n = 1,...,res_y``.
    """

    x = np.linspace(xmin, xmax, res_x)
    y = np.linspace(ymin, ymax, res_y)
    X,Y = np.meshgrid(x, y)
    Z = X + Y*1j
    return Z


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_real(
        x: np.ndarray,
        fx: np.ndarray,
        fstr: str,
        lw: float = 1.75,
    ) -> plt.Figure:
    """
    Plot ``fx = f(x)`` given a real (scalar-valued) function `f` and data-points `x`

    Parameters
    ----------
    x : 1D np.ndarray
        Horizontal coordinates of data-points
    fx : 1D np.ndarray
        Values of the function `f` at data-points `x`
    fstr : str
        Human-readable descriptor of the function `f` (used as title)
    lw : float
        Line-width used to plot the function

    Returns
    -------
    fig : plt.Figure
        Figure containing the plot
    """

    fig, ax = plt.subplots()
    ax.plot(x, fx, lw=lw)
    ax.axhline(y=0, color="black", lw=0.7*lw)
    ax.axvline(x=0, color="black", lw=0.7*lw)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title(f"Die Funktion {fstr} in $\mathbb{{R}}$")
    plt.show()
    return fig


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_complex(
        fz: np.ndarray,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        fstr: str,
        cmap: str = "inferno_r",
        interp: str = "bilinear"
    ) -> plt.Figure:
    """
    Render an amplitude/phase plot of the complex array `fz`

    Code is largely based on https://stackoverflow.com/a/20958684

    Parameters
    ----------
    fz : 2D np.ndarray
        Array of complex numbers
    xmin : float
        Minimal value of real part ``Re(fz)`` (used for axis labeling)
    xmax : float
        Maximal value of real part ``Re(fz)`` (used for axis labeling)
    ymin : float
        Minimal value of imaginary part ``Im(fz)`` (used for axis labeling)
    ymax : float
        Maximal value of imaginary part ``Im(fz)`` (used for axis labeling)
    fstr : str
        Human-readable descriptor of the array `fz` (used as title)
    cmap : str
        Name of colormap to be used for rendering `fz` (for available options
        see https://matplotlib.org/stable/tutorials/colors/colormaps.html)
    interp : str
        Interpolation method to be used for rendering `fz` (for available options
        see https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html)

    Returns
    -------
    fig : plt.Figure
        Figure containing the plot
    """

    r = np.abs(fz)
    arg = np.angle(fz)
    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0 / (1.0 + r**0.3)
    s = 0.8
    c = np.vectorize(hls_to_rgb) (h,l,s)
    c = np.array(c)
    c = c.transpose(1,2,0)
    fig, ax = plt.subplots()
    ax.imshow(c, cmap=cmap, origin="lower", vmin=c.min(), vmax=c.max(),
              interpolation=interp, extent=[xmin, xmax, ymin, ymax])
    ax.set_xlabel("$\Re(z)$")
    ax.set_ylabel("$\Im(z)$")
    fzstr = fstr.replace("x", "z")
    ax.set_title(f"Die Funktion {fzstr} in $\mathbb{{C}}$")
    plt.show()
    return fig


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_fractal1d(
        x: np.ndarray,
        fx: np.ndarray,
        frac: np.ndarray,
        fstr: str,
        cmap: str = "plasma_r",
        lw: float = 2.0
        ) -> plt.Figure:
    """
    Render a 1D Newton fractal

    Parameters
    ----------
    x : 1D np.ndarray
        Horizontal coordinates of data-points
    fx : 1D np.ndarray
        Values of the function `f` at data-points `x`
    frac : 1D np.ndarray
        Newton fractal values (between 0 and 1)
    fstr : str
        Human-readable descriptor of the function `f` (used as title)
    cmap : str
        Name of colormap to be used for rendering (for available options
        see https://matplotlib.org/stable/tutorials/colors/colormaps.html)
    lw : float
        Line-width used to plot the function `f`

    Returns
    -------
    fig : plt.Figure
        Figure containing the plot
    """

    fig, ax = plt.subplots()
    fmin = frac.min()
    fmax = frac.max()
    norm = plt.Normalize(fmin, fmax)
    ax.plot(x, fx, lw=lw)
    for i in range(x.size - 1):
        ax.fill_between([x[i], x[i+1]], [fx[i], fx[i+1]], color=mpl.cm.__dict__[cmap](norm(frac[i])))
    ax.axhline(y=0, color="black", lw=1.1*lw)
    ax.axvline(x=0, color="black", lw=1.1*lw)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title(f"Konvergenz mit {fstr} in $\mathbb{{R}}$")
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), norm=norm, ax=ax, ticks=[fmin, (fmin + fmax)/2.0, fmax])
    cbar.ax.set_yticklabels(["schlecht", "okay", "gut"])
    plt.show()
    return fig


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_fractal2d(
        frac: np.ndarray,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        fstr: str,
        cmap: str = "inferno_r",
        interp: str = "bilinear",
        ) -> plt.Figure:
    """
    Render the Newton fractal of a complex-valued function ``f(z)``

    Parameters
    ----------
    frac : 2D np.ndarray
        Newton fractal values (between 0 and 1)
    xmin : float
        Minimal value of real part ``Re(f(z))`` (used for axis labeling)
    xmax : float
        Maximal value of real part ``Re(f(z))`` (used for axis labeling)
    ymin : float
        Minimal value of imaginary part ``Im(f(z))`` (used for axis labeling)
    ymax : float
        Maximal value of imaginary part ``Im(f(z))`` (used for axis labeling)
    fstr : str
        Human-readable descriptor of the function ``f(z)`` (used as title)
    cmap : str
        Name of colormap to be used for rendering ``f(z)`` (for available options
        see https://matplotlib.org/stable/tutorials/colors/colormaps.html)
    interp : str
        Interpolation method to be used for rendering ``f(z)`` (for available options
        see https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html)

    Returns
    -------
    fig : plt.Figure
        Figure containing the plot
    """

    fig, ax = plt.subplots()
    fmin = frac.min()
    fmax = frac.max()
    ax.imshow(frac, cmap=cmap, origin="lower", vmin=fmin, vmax=fmax,
              interpolation=interp, extent=[xmin, xmax, ymin, ymax])
    ax.set_xlabel("$\Re(z)$")
    ax.set_ylabel("$\Im(z)$")
    fzstr = fstr.replace("x", "z")
    ax.set_title(f"Konvergenz mit {fzstr} in $\mathbb{{C}}$")
    norm = plt.Normalize(fmin, fmax)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), norm=norm, ax=ax, ticks=[fmin, (fmin + fmax)/2.0, fmax])
    cbar.ax.set_yticklabels(["schlecht", "okay", "gut"])
    plt.show()
    return fig


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def compute_fractals(
        f: Callable[[np.ndarray], np.ndarray],
        df: Callable[[np.ndarray], np.ndarray],
        z0: np.ndarray,
        c0: float,
        domain: list,
        fstr: str,
        idx: int,
        outfolder: str,
        cmap: str = "inferno_r",
    ) -> np.ndarray:
    """
    Compute and render Newton fractal of the linearly parameterized
    function ``f(z,c) = F(z) + c`` for a given fixed tuple ``(z0, c0)``

    Parameters
    ----------
    f : callable
        Parameterized function ``f(z,c)``. Must accept a complex 2D array `z`
        and real-valued 1D array `c`.
    df : callable
        Derivative ``df(z,c)/dz = dF(z)/dz`. Must accept a complex 2D array `z`.
    z0 : 2D np.ndarray
        Initial value(s) for Newton iteration.
    c0 : float
        Value of parameter `c`
    domain : list
        Min/max values of real/imaginary parts of ``f(z,c)`` with `z` in
        the domain of interest, i.e.,
        ``domain = [max(Re(f)), min(Re(f)), max(Im(f)), min(Im(f))]``
    fstr : str
        Human-readable descriptor of the function ``f(z,c)`` (used as title)
    idx : int
        Running index corresponding to the discretization of `c`
    outfolder : str
        Path to a directory for storing generated fractal renderings (must
        already exist)
    cmap : str
        Name of colormap to be used for rendering ``f(z,c)`` (for available options
        see https://matplotlib.org/stable/tutorials/colors/colormaps.html)

    Returns
    -------
    frac : 2D np.ndarray
        Iteration number `k` normalized by `max_iter` as returned by
        :func:`newton_method`
    """

    fc0 = lambda z: f(z, c0)
    conv = newton_method(fc0, df, z0)
    fig = plot_fractal2d(conv, *domain, fstr.format(c0=c0), cmap=cmap)
    fig.savefig(os.path.abspath(os.path.expanduser(os.path.join(outfolder, f"{idx}.png"))),
                dpi=300)
    plt.close(fig)
    return conv
