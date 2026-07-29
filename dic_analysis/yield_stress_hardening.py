#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 of the pixel-wise DIC pipeline: yield stress and hardening identification.

Consumes the ``U_<i>.npy`` / ``V_<i>.npy`` displacement fields produced by
``dic_displacement_fields.py`` and identifies, pixel by pixel:

  * the von Mises equivalent strain history (Evm) from the deviatoric strain,
  * the yield onset frame (S_p), detected as the end of the linear elastic
    regime via an R2 criterion on the Evm-vs-load-step curve,
  * the yield stress at that onset (Hooke and deviatoric estimates),
  * the local hardening coefficient K, normalised by the macroscopic slope.

Original author: Qi Hu - restructured by kilincadil
"""

# ============================================================
# 0. IMPORTS
# ============================================================

import os
import re

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed

plt.ion()


# ============================================================
# CONFIG
# ============================================================

# --- Paths -----------------------------------------------------------------
FOLDER_U = r"C:\Users\adil.kilinc\Desktop\Thesis\3_data\21_DIC\U_a100"
FOLDER_V = r"C:\Users\adil.kilinc\Desktop\Thesis\3_data\21_DIC\V_a100"
OUT_DIR = (r"C:\Users\adil.kilinc\Desktop\Thesis\3_data"
           r"\27_yield_hardening_Adil\Main_a100")

# --- Material properties ---------------------------------------------------
E = 205000.0    # Young's modulus [MPa]
NU = 0.3        # Poisson's ratio [-]

# --- Analysis parameters ---------------------------------------------------
# Load steps whose Evm fields are corrupted and replaced by an interpolation
# between neighbouring steps. Format: {bad_index: (index_before, index_after)}.
CORRUPTED_FRAMES = {31: (29, 33), 32: (29, 35)}

# Temporal smoothing of the Evm stack (sigma along the load-step axis only).
GAUSSIAN_SIGMA = [1, 0, 0]

# Valid range of yield-onset frame indices; pixels outside are left as NaN.
S_P_MIN = 2
S_P_MAX = 42

# Reference hardening slope taken from the macroscopic tensile curve.
SLOPE_REF = 4.46e-05

# Plausible range for the hardening coefficient; outliers are masked to NaN
# and then filled from their neighbours.
HARDENING_MIN = 0.0
HARDENING_MAX = 3.0

N_JOBS = -1  # joblib workers (-1 = all cores)


# ============================================================
# 1. NUMERICAL & REGRESSION TOOLS
# ============================================================

@njit
def estimate_coef(x, y):
    """Ordinary least-squares fit of y = intercept + slope * x."""
    n = x.shape[0]
    m_x = np.mean(x)
    m_y = np.mean(y)
    ss_xy = np.sum(y * x) - n * m_y * m_x
    ss_xx = np.sum(x * x) - n * m_x * m_x
    slope = ss_xy / ss_xx
    intercept = m_y - slope * m_x
    return intercept, slope


@njit
def fct_slope(u, v, Evm, S_p, len_x):
    """Post-yield slope of pixel (u, v) after removing the elastic trend."""
    if 0 < S_p[u, v] < Evm.shape[0]:
        y_true = Evm[:, u, v]
        k = int(S_p[u, v])
        intercept, slope = estimate_coef(len_x[:k], y_true[:k])
        y_mean_pred = np.linspace(1, Evm.shape[0], Evm.shape[0])
        y_mean_pred[:] = intercept + slope * len_x[:Evm.shape[0]]
        y_mean_true_plastic = y_true - y_mean_pred
        try:
            _, slope_p = estimate_coef(
                len_x[k:Evm.shape[0]],
                y_mean_true_plastic[k:Evm.shape[0]],
            )
        except Exception:
            slope_p = np.nan
    else:
        slope_p = np.nan
    return slope_p


@njit
def fct_slope_negative(u, v, Evm, S_p, len_x):
    """Fallback slope: raw post-yield fit, used where fct_slope is negative."""
    slope = np.nan
    if 0 < S_p[u, v] < Evm.shape[0]:
        y_true = Evm[:, u, v]
        k = int(S_p[u, v])
        _, slope = estimate_coef(len_x[k:], y_true[k:])
    return slope


def replace_nan_with_neighbors(a, max_iter=64):
    """Iterative vectorised 8-neighbour NaN filling.

    Handles clusters of NaNs by repeating until none remain or max_iter reached.
    """
    a = a.astype(float, copy=True)
    for _ in range(max_iter):
        nan = np.isnan(a)
        if not nan.any():
            break
        neighbors = [
            np.pad(a, ((1, 0), (1, 0)), constant_values=np.nan)[:-1, :-1],  # top-left
            np.pad(a, ((1, 0), (0, 0)), constant_values=np.nan)[:-1, :],    # top
            np.pad(a, ((1, 0), (0, 1)), constant_values=np.nan)[:-1, 1:],   # top-right
            np.pad(a, ((0, 0), (1, 0)), constant_values=np.nan)[:, :-1],    # left
            np.pad(a, ((0, 0), (0, 1)), constant_values=np.nan)[:, 1:],     # right
            np.pad(a, ((0, 1), (1, 0)), constant_values=np.nan)[1:, :-1],   # bottom-left
            np.pad(a, ((0, 1), (0, 0)), constant_values=np.nan)[1:, :],     # bottom
            np.pad(a, ((0, 1), (0, 1)), constant_values=np.nan)[1:, 1:],    # bottom-right
        ]
        stack = np.stack(neighbors, axis=0)
        sums = np.nansum(stack, axis=0)
        counts = np.sum(np.isfinite(stack), axis=0)
        means = sums / np.where(counts == 0, 1, counts)
        a[nan & (counts > 0)] = means[nan & (counts > 0)]
    return a


def get_sorted_npy_files(folder, prefix):
    """List ``<prefix>_<n>.npy`` files in *folder*, sorted by their index n."""
    files = [f for f in os.listdir(folder) if re.match(rf'{prefix}_\d+\.npy$', f)]
    return sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))


# ============================================================
# 2. STRAIN & STRESS RELATIONS
# ============================================================

def deviatoric_strain_plane_stress(exx, eyy, exy, nu):
    """Deviatoric strain components under a plane-stress assumption."""
    ezz = -nu / (1.0 - nu) * (exx + eyy)
    trace = exx + eyy + ezz
    exx_dev = exx - trace / 3.0
    eyy_dev = eyy - trace / 3.0
    ezz_dev = ezz - trace / 3.0
    exy_dev = exy
    return exx_dev, eyy_dev, ezz_dev, exy_dev


def evm_from_edev(exx_dev, eyy_dev, ezz_dev, exy_dev):
    """von Mises equivalent strain from deviatoric strain components."""
    j2 = exx_dev**2 + eyy_dev**2 + ezz_dev**2 + 2.0 * exy_dev**2
    return np.sqrt((2.0 / 3.0) * j2)


def vm_Jeff(E, nu, exx, eyy, exy, evm):
    """von Mises stress, two estimates.

    Returns
    -------
    sigma_vm_hooke : ndarray
        From plane-stress Hooke's law on the total strain.
    sigma_vm_dev : ndarray
        From the deviatoric equivalent strain, sigma = 3 * G * evm.
    """
    pref = E / (1.0 - nu**2)
    sxx = pref * (exx + nu * eyy)
    syy = pref * (eyy + nu * exx)
    g = E / (2.0 * (1.0 + nu))
    sxy = 2.0 * g * exy
    sigma_vm_hooke = np.sqrt(sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2)
    sigma_vm_dev = 3.0 * g * evm
    return sigma_vm_hooke, sigma_vm_dev


def strain_components(u_field, v_field, nu):
    """Strain and deviatoric-equivalent strain from a displacement field pair."""
    exy, exx = np.gradient(u_field)
    eyy, eyx = np.gradient(v_field)
    eshear = 0.5 * (exy + eyx)
    exx_dev, eyy_dev, ezz_dev, exy_dev = deviatoric_strain_plane_stress(
        exx, eyy, eshear, nu
    )
    evm_dev = evm_from_edev(exx_dev, eyy_dev, ezz_dev, exy_dev)
    return exx, eyy, eshear, evm_dev


# ============================================================
# 3. YIELD ONSET DETECTION (R2-BASED)
# ============================================================

@njit
def find_local_maxima(arr):
    """Indices of strictly increasing-then-decreasing peaks (2-point window)."""
    local_maxima = []
    for i in range(2, len(arr) - 2):
        if (arr[i] > arr[i - 1] and arr[i - 1] > arr[i - 2] and
                arr[i] > arr[i + 1] and arr[i + 1] > arr[i + 2]):
            local_maxima.append(i)
    return local_maxima


@njit
def detect_maximum_r2(u, v, Evm, len_x):
    """Yield onset frame for pixel (u, v).

    Fits the Evm history over a growing window and returns the frame at which
    the linear (elastic) fit stops improving, i.e. the first local maximum of
    R2, falling back to the global maximum if no local peak exists.
    """
    evm_uv = Evm[:, u, v]
    b = []
    n = Evm.shape[0]
    for k in range(5, n):
        intercept, slope = estimate_coef(len_x[3:k], evm_uv[3:k])
        if slope > -100:
            y_mean_pred = np.linspace(1, n, n)
            y_mean_pred[:] = intercept + slope * len_x[:n]
            try:
                corr_matrix = np.corrcoef(evm_uv[:k], y_mean_pred[:k])
                corr = corr_matrix[0, 1]
                b.append(corr**2)
            except Exception:
                b.append(-10000)
        else:
            b.append(-10000)

    try:
        l = find_local_maxima(np.array(b[1:]))[0] + 6
    except Exception:
        max_value = max(b[1:])
        l = b.index(max_value) + 6

    # Push the onset forward while the elastic fit still has a non-positive slope.
    for w in range(l, n):
        _, slope = estimate_coef(len_x[:w], evm_uv[:w])
        if slope <= 0:
            l += 1
    return l


def compute_yield_onset_map(Evm, len_x, verbose=True):
    """Yield onset frame S_p for every pixel."""
    S_p = np.full((Evm.shape[1], Evm.shape[2]), np.nan)
    for i in range(Evm.shape[1]):
        if verbose and i % 100 == 0:
            print(f"  yield onset: row {i}/{Evm.shape[1]}")
        for j in range(Evm.shape[2]):
            S_p[i, j] = detect_maximum_r2(i, j, Evm, len_x)
    return S_p


# ============================================================
# 4. YIELD STRESS MAP
# ============================================================

def compute_stress_yield_map(S_p, folder_U, folder_V, files_U, files_V,
                             E=E, nu=NU, n_jobs=N_JOBS):
    """Yield stress at each pixel, sampled at that pixel's yield onset frame.

    Returns
    -------
    (stress_hooke, stress_dev) : tuple of ndarray
        Hooke-based and deviatoric-based von Mises yield stress maps [MPa].
    """
    stress_hooke = np.full(S_p.shape, np.nan, dtype=float)
    stress_dev = np.full(S_p.shape, np.nan, dtype=float)

    U_all = [np.load(os.path.join(folder_U, f), mmap_mode='r') for f in files_U]
    V_all = [np.load(os.path.join(folder_V, f), mmap_mode='r') for f in files_V]

    valid = (S_p >= S_P_MIN) & (S_p <= S_P_MAX) & ~np.isnan(S_p)
    m_map = np.full(S_p.shape, -1, dtype=int)
    m_map[valid] = S_p[valid].astype(int)

    def process_m(m):
        exx, eyy, eshear, evm_dev = strain_components(U_all[m], V_all[m], nu)
        sigma_hooke_m, sigma_dev_m = vm_Jeff(E, nu, exx, eyy, eshear, evm_dev)
        return m, sigma_hooke_m, sigma_dev_m

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_m)(m) for m in range(len(files_U))
    )

    for m, sigma_hooke_m, sigma_dev_m in results:
        idx = np.argwhere(m_map == m)
        if len(idx) == 0:
            continue
        i_coords, j_coords = idx[:, 0], idx[:, 1]
        stress_hooke[i_coords, j_coords] = sigma_hooke_m[i_coords, j_coords]
        stress_dev[i_coords, j_coords] = sigma_dev_m[i_coords, j_coords]

    return stress_hooke, stress_dev


# ============================================================
# 5. HARDENING SLOPE MAP
# ============================================================

@njit(parallel=True)
def compute_slope_plastic(Evm, S_p, len_x):
    """Post-yield (plastic) slope map."""
    n1, n2 = Evm.shape[1], Evm.shape[2]
    slope_plastic = np.full((n1, n2), np.nan)
    for i in prange(n1):
        for j in range(n2):
            slope_plastic[i, j] = fct_slope(i, j, Evm, S_p, len_x)
    return slope_plastic


@njit(parallel=True)
def update_slope_plastic(Evm, slope_plastic, S_p, len_x):
    """Recompute negative plastic slopes with the raw post-yield fit."""
    n1, n2 = Evm.shape[1], Evm.shape[2]
    for i in prange(n1):
        for j in range(n2):
            if slope_plastic[i, j] < 0:
                slope_plastic[i, j] = fct_slope_negative(i, j, Evm, S_p, len_x)
    return slope_plastic


# ============================================================
# 6. PLOTTING
# ============================================================

def plot_hardening_map(hardening_coef_filled,
                       vmin=HARDENING_MIN, vmax=HARDENING_MAX):
    """Display the hardening coefficient map."""
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(8, 7))
    plt.imshow(hardening_coef_filled, plt.get_cmap('jet'), alpha=1)
    plt.colorbar()
    plt.clim(vmin, vmax)
    plt.axis('off')
    fig.tight_layout()
    plt.show()
    return fig


# ============================================================
# 7. MAIN WORKFLOW
# ============================================================

def main():
    files_U = get_sorted_npy_files(FOLDER_U, 'U')
    files_V = get_sorted_npy_files(FOLDER_V, 'V')
    if not files_U or len(files_U) != len(files_V):
        raise RuntimeError(
            f"Mismatched or missing field files: {len(files_U)} U, {len(files_V)} V"
        )
    print(f"Found {len(files_U)} load steps.")

    U_all = [np.load(os.path.join(FOLDER_U, f), mmap_mode='r') for f in files_U]
    V_all = [np.load(os.path.join(FOLDER_V, f), mmap_mode='r') for f in files_V]

    # ---------- PART I: Evm history ----------
    print("Part I: equivalent strain history...")
    Evm_all_dev = [
        strain_components(U_all[m], V_all[m], NU)[3] for m in range(len(U_all))
    ]
    Evm = np.array(Evm_all_dev)

    for bad, (before, after) in CORRUPTED_FRAMES.items():
        Evm[bad] = (Evm[before] + Evm[after]) / 2

    Evm = gaussian_filter(Evm, sigma=GAUSSIAN_SIGMA)
    len_x = np.linspace(1, Evm.shape[0], Evm.shape[0])

    # ---------- PART II.1: yield onset ----------
    print("Part II.1: yield onset detection...")
    S_p = compute_yield_onset_map(Evm, len_x)

    # ---------- PART II.2: yield stress ----------
    print("Part II.2: yield stress...")
    stress_yield_hooke, stress_yield_dev = compute_stress_yield_map(
        S_p=S_p,
        folder_U=FOLDER_U,
        folder_V=FOLDER_V,
        files_U=files_U,
        files_V=files_V,
        E=E,
        nu=NU,
        n_jobs=N_JOBS,
    )

    # ---------- PART III: hardening coefficient ----------
    print("Part III: hardening coefficient...")
    slope_plastic = compute_slope_plastic(Evm, S_p, len_x)
    slope_plastic = update_slope_plastic(Evm, slope_plastic, S_p, len_x)

    hardening_coef = SLOPE_REF / slope_plastic
    hardening_masked = hardening_coef.copy()
    hardening_masked[hardening_masked > HARDENING_MAX] = np.nan
    hardening_masked[hardening_masked < HARDENING_MIN] = np.nan
    hardening_coef_filled = replace_nan_with_neighbors(hardening_masked)

    plot_hardening_map(hardening_coef_filled)

    # ---------- Save ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "S_p_yield_onset_frame.npy"), S_p)
    np.save(os.path.join(OUT_DIR, "yield_stress_vm.npy"), stress_yield_hooke)
    np.save(os.path.join(OUT_DIR, "yield_stress_dev.npy"), stress_yield_dev)
    np.save(os.path.join(OUT_DIR, "K_hardening_coef_filled.npy"),
            hardening_coef_filled)
    print(f"Done. Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
