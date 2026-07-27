# -*- coding: utf-8 -*-
"""
seam_metrics.py – interface equilibrium residual for domain-decomposed FE
(the diagnostic used to calibrate how much overlap padding a partitioned /
tiled solve needs before stitching independently-solved tiles back together).

Theory: the converged monolithic FE solution satisfies nodal force
equilibrium: the assembled internal forces f_int = sum_e (int B^T sigma dV)
cancel at every free node. Stitching cores of independently solved tiles
breaks this cancellation at the seams; the assembled nodal residual there IS
the partitioning error (interface residual of an overlapping Schwarz step).

Note on the floor: computed from ELEMENT-MEAN stresses, even the monolithic
solution has a small nonzero residual (Gauss-point variation is lost). Always
compare seam values against the far-from-seam floor / monolithic reference
computed with the SAME post-processing.

Normalization: ref_mag = mean norm of the per-node element force
contributions (Abaqus-style reference force magnitude).

Requires fem_pixel.py (same package).
"""
import numpy as np
from fem_pixel import _B_detJ, GP_XI, GP_W


def _geometry(nx, ny, element_size, scale_factor):
    """Uniform-mesh assembly data: Q (8x3) with f_e = Q @ sigma_e, plus the
    element->dof scatter map ld (n_e, 8). Element order matches S[i, j]
    flattened in C order."""
    el = element_size * scale_factor
    c = np.array([[0., 0.], [el, 0.], [el, el], [0., el]])
    Q = np.zeros((8, 3))
    for (xi, eta), w in zip(GP_XI, GP_W):
        B, dJ = _B_detJ(c, xi, eta)
        Q += w * dJ * B.T
    node_ids = np.arange((nx + 1) * (ny + 1)).reshape((nx + 1, ny + 1), order='F')
    ie, je = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    conn = np.stack([node_ids[ie, je], node_ids[ie + 1, je],
                     node_ids[ie + 1, je + 1], node_ids[ie, je + 1]],
                    axis=-1).reshape(-1, 4)                    # C order over (i, j)
    ld = np.empty((nx * ny, 8), dtype=int)
    ld[:, 0::2] = 2 * conn
    ld[:, 1::2] = 2 * conn + 1
    return Q, ld, node_ids


def nodal_residual(S, element_size, scale_factor):
    """Assembled nodal internal-force residual from an element stress grid
    S (nx, ny, 3) [s11, s22, s12].
    Returns R (nx+1, ny+1, 2) nodal force vectors and ref_mag (scalar)."""
    nx, ny, _ = S.shape
    Q, ld, node_ids = _geometry(nx, ny, element_size, scale_factor)
    fe = S.reshape(-1, 3) @ Q.T                                # (n_e, 8)
    F = np.zeros(2 * (nx + 1) * (ny + 1))
    np.add.at(F, ld, fe)
    R = np.stack([F[2 * node_ids], F[2 * node_ids + 1]], axis=-1)
    contrib = np.stack([fe[:, 0::2], fe[:, 1::2]], axis=-1)    # (n_e, 4, 2)
    ref_mag = float(np.mean(np.linalg.norm(contrib, axis=-1)))
    return R, ref_mag


def seam_profile(R, ref_mag, seams_x, seams_y, d_max=60):
    """Residual magnitude vs distance to the nearest seam (node lines).
    seams_x / seams_y: node indices of vertical / horizontal seams.
    Returns dict with:
      d          : distances 0..d_max
      profile    : mean |R|/ref per distance bucket
      seam_max   : max |R|/ref on seam nodes (d=0)
      seam_l2    : rms |R|/ref on seam nodes
      floor      : mean |R|/ref at d >= d_max (far-field floor)
    Outer boundary nodes are excluded (they carry reactions, not residuals)."""
    mag = np.linalg.norm(R, axis=-1) / max(ref_mag, 1e-30)
    nxn, nyn = mag.shape
    interior = np.zeros_like(mag, dtype=bool)
    interior[1:-1, 1:-1] = True

    ii = np.arange(nxn)[:, None]
    jj = np.arange(nyn)[None, :]
    d = np.full(mag.shape, np.inf)
    for s in seams_x:
        d = np.minimum(d, np.abs(ii - s) * np.ones_like(jj, dtype=float))
    for s in seams_y:
        d = np.minimum(d, np.ones_like(ii, dtype=float) * np.abs(jj - s))
    if not (len(seams_x) or len(seams_y)):
        raise ValueError("no seams given")

    dists = np.arange(d_max + 1)
    profile = np.array([np.nanmean(mag[interior & (d == k)]) if
                        np.any(interior & (d == k)) else np.nan for k in dists])
    on_seam = interior & (d == 0)
    far = interior & (d >= d_max)
    return dict(d=dists,
                profile=profile,
                seam_max=float(np.nanmax(mag[on_seam])),
                seam_l2=float(np.sqrt(np.nanmean(mag[on_seam] ** 2))),
                floor=float(np.nanmean(mag[far])) if np.any(far) else np.nan)


def displacement_jumps(edges, npx, npy):
    """Displacement discontinuity across seams BEFORE stitching discards one
    side. edges[(i,j)] = dict(L,R,B,T) core-edge nodal displacement lines
    (n_nodes, 2), from each tile's own solution.
    Returns dict with max and rms jump (same units as u; mm here).
    Note: with PAD=0 the seam nodes are prescribed BCs of both tiles, so the
    jump is exactly 0 by construction - the force residual is the signal there."""
    jumps = []
    for i in range(npx - 1):
        for j in range(npy):
            jumps.append(np.linalg.norm(edges[(i, j)]["R"] - edges[(i + 1, j)]["L"], axis=-1))
    for i in range(npx):
        for j in range(npy - 1):
            jumps.append(np.linalg.norm(edges[(i, j)]["T"] - edges[(i, j + 1)]["B"], axis=-1))
    if not jumps:
        return dict(max=0.0, rms=0.0)
    allj = np.concatenate(jumps)
    return dict(max=float(allj.max()), rms=float(np.sqrt(np.mean(allj ** 2))))
