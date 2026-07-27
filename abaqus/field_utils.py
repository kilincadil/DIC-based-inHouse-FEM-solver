# -*- coding: utf-8 -*-
"""
Array helpers shared by the .inp generation drivers.

Everything here operates on ``[i, j]`` arrays with ``i`` along x and ``j``
along y, the convention used throughout :mod:`inp_writer`.

Part of: DIC-based in-house FEM (https://github.com/kilincadil/DIC-based-inHouse-FEM)
"""

import numpy as np


def hardening_table(yield_map, K_map, n_exp, ep_max=0.2, n_points=50):
    """Tabulate a Ludwik flow curve ``sy(ep) = sy0 + K * ep**n`` per element.

    Produces exactly the table Abaqus interpolates under ``*Plastic``, sampled
    on a uniform plastic-strain grid.

    Parameters
    ----------
    yield_map, K_map : ndarray, shape (nx_elems, ny_elems)
        Per-element initial yield stress (MPa) and hardening coefficient.
    n_exp : float
        Ludwik hardening exponent.
    ep_max : float, optional
        Upper bound of the plastic-strain grid. Must match
        ``inp_writer.PLASTIC_STRAIN_MAX``.
    n_points : int, optional
        Number of table rows. Must match ``inp_writer.PLASTIC_TABLE_POINTS``.

    Returns
    -------
    ndarray, shape (n_points, nx_elems, ny_elems)
        Flow stress at each tabulated plastic strain, for every element.
    """
    yield_map = np.asarray(yield_map, dtype=float)
    K_map = np.asarray(K_map, dtype=float)

    if yield_map.shape != K_map.shape:
        raise ValueError("yield_map {0} and K_map {1} must have the same shape".format(
            yield_map.shape, K_map.shape))

    ep = np.linspace(0.0, ep_max, n_points)
    # ep[:, None, None] broadcasts the strain grid against the (nx, ny) maps.
    return yield_map[None, :, :] + K_map[None, :, :] * ep[:, None, None] ** n_exp


def crop_center(array, nx, ny):
    """Return the central ``(nx, ny)`` block of a 2-D array.

    Note the argument order: ``nx`` cuts the **first** axis and ``ny`` the
    second, consistent with the ``[i, j] = [x, y]`` convention. Passing them the
    other way round silently produces a transposed crop on non-square domains.
    """
    if array.shape[0] < nx or array.shape[1] < ny:
        raise ValueError("cannot crop {0} out of an array of shape {1}".format(
            (nx, ny), array.shape))

    i0 = (array.shape[0] - nx) // 2
    j0 = (array.shape[1] - ny) // 2
    return array[i0:i0 + nx, j0:j0 + ny]


def fill_nan_iter_4nbr(a, max_iter=64):
    """Fill NaNs by repeated averaging over finite 4-neighbours.

    DIC fields typically carry NaNs where correlation failed. Each sweep
    replaces every NaN that has at least one finite orthogonal neighbour by the
    mean of those neighbours, so holes are filled inward from their edges.
    Returns early once no NaN remains; NaNs in regions with no finite neighbour
    within ``max_iter`` sweeps survive.
    """
    a = np.asarray(a).astype(float, copy=True)

    for _ in range(max_iter):
        nan = np.isnan(a)
        if not nan.any():
            break

        up = np.pad(a, ((1, 0), (0, 0)), mode="constant", constant_values=np.nan)[:-1, :]
        down = np.pad(a, ((0, 1), (0, 0)), mode="constant", constant_values=np.nan)[1:, :]
        left = np.pad(a, ((0, 0), (1, 0)), mode="constant", constant_values=np.nan)[:, :-1]
        right = np.pad(a, ((0, 0), (0, 1)), mode="constant", constant_values=np.nan)[:, 1:]

        stack = np.stack([up, down, left, right], axis=0)
        counts = np.sum(np.isfinite(stack), axis=0)
        means = np.nansum(stack, axis=0) / np.where(counts == 0, 1, counts)

        fillable = nan & (counts > 0)
        a[fillable] = means[fillable]

    return a


def pad_last_row_col(a):
    """Extend an array by one row and one column, repeating the edge values.

    A field sampled per element has one fewer node than element along each
    axis; this pads an element-shaped array up to the matching node grid.
    """
    return np.pad(a, ((0, 1), (0, 1)), mode="edge")


def element_count_label(nx_elems, ny_elems):
    """Return a compact element-count tag for file names, e.g. ``446k``, ``2m``."""
    n = int(nx_elems) * int(ny_elems)

    if n >= 1_000_000:
        return "{0}m".format(n // 1_000_000)
    if n >= 1_000:
        return "{0}k".format(n // 1_000)
    return str(n)
