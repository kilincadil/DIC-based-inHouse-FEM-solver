#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 of the pixel-wise DIC pipeline: displacement field computation.

Computes dense displacement fields (U, V) between a reference image and each
image of a tensile (traction) sequence using OpenCV's DIS optical flow with
variational refinement.

Outputs one ``U_<i>.npy`` / ``V_<i>.npy`` pair per loading step, cropped to the
region of interest. These files are the input of ``yield_stress_hardening.py``.

Original author: Eddidoune (2021)
Adapted by: kilincadil
"""

import os
import sys
import shutil
from glob import glob

import numpy as np

# ============================================================
# CONFIG
# ============================================================

# --- Custom OpenCV build ---------------------------------------------------
# The DIS optical flow variational-refinement setters used below are only
# exposed by a locally built OpenCV. Set USE_CUSTOM_OPENCV = False to fall back
# to the pip-installed cv2.
USE_CUSTOM_OPENCV = True
OPENCV_LIB_DIR = r"C:\Users\adil.kilinc\opencv\build\lib\python3"
OPENCV_BIN_DIR = r"C:\Users\adil.kilinc\opencv\build\bin"
OPENCV_PYD_NAME = "cv2.cp312-win_amd64.pyd"

# --- Data paths ------------------------------------------------------------
MAIN_PATH = r"C:\Users\adil.kilinc\Desktop\Thesis\3_data\21_DIC"
REF_IMAGE = os.path.join(MAIN_PATH, "ref.tif")
MASK_IMAGE = os.path.join(MAIN_PATH, "mask.png")
TRACTION_DIR = os.path.join(MAIN_PATH, "traction")
OUT_DIR_U = os.path.join(MAIN_PATH, "U_a100")
OUT_DIR_V = os.path.join(MAIN_PATH, "V_a100")

# --- Optical flow parameters ----------------------------------------------
# Variational refinement, cf. Brox et al., ECCV 2004:
# https://www.mia.uni-saarland.de/Publications/brox-eccv04-of.pdf
ALPHA = 100.0       # smoothness weight
DELTA = 1.0         # colour/greyscale constancy weight
GAMMA = 0.0         # gradient constancy weight
EPSILON = 0.002     # convergence threshold
ITERATIONS = 30     # variational refinement iterations
FINEST_SCALE = 0
PATCH_SIZE = 4
PATCH_STRIDE = 1

# --- Region of interest (row_start, row_stop, col_start, col_stop) ---------
ROI = (400, 4000, 1211, 4311)


# ============================================================
# OPENCV IMPORT
# ============================================================

def _load_custom_opencv(lib_dir, bin_dir, pyd_name):
    """Make a locally built OpenCV importable, then import it."""
    pyd_src = os.path.join(lib_dir, pyd_name)
    pyd_dst = os.path.join(lib_dir, "cv2.pyd")
    if not os.path.exists(pyd_dst):
        shutil.copyfile(pyd_src, pyd_dst)
    os.add_dll_directory(bin_dir)
    sys.path.insert(0, lib_dir)


if USE_CUSTOM_OPENCV:
    _load_custom_opencv(OPENCV_LIB_DIR, OPENCV_BIN_DIR, OPENCV_PYD_NAME)

import cv2  # noqa: E402  (import must follow the sys.path setup above)


# ============================================================
# CORE
# ============================================================

def dic_displacement(im1, im2, alpha=10.0, delta=1.0, gamma=0.0,
                     epsilon=0.05, iterations=30):
    """Dense DIS optical flow between two greyscale images.

    Parameters
    ----------
    im1, im2 : ndarray
        Reference and deformed greyscale images (same shape, uint8).
    alpha, delta, gamma, epsilon, iterations
        Variational refinement parameters (see module CONFIG).

    Returns
    -------
    (U, V) : tuple of ndarray
        X- and Y-displacement fields, same shape as the input images.
    """
    flow = cv2.DISOpticalFlow_create()
    flow.setFinestScale(FINEST_SCALE)
    flow.setVariationalRefinementAlpha(alpha)
    flow.setVariationalRefinementDelta(delta)
    flow.setVariationalRefinementGamma(gamma)
    flow.setVariationalRefinementEpsilon(epsilon)
    flow.setVariationalRefinementIterations(iterations)
    flow.setPatchSize(PATCH_SIZE)
    flow.setPatchStride(PATCH_STRIDE)

    res = flow.calc(im1, im2, None)
    u = res[:, :, 0]  # X-displacement
    v = res[:, :, 1]  # Y-displacement
    return u, v


def main():
    print(f"OpenCV: {cv2.__file__} (version {cv2.__version__})")

    os.makedirs(OUT_DIR_U, exist_ok=True)
    os.makedirs(OUT_DIR_V, exist_ok=True)

    im_ref = cv2.imread(REF_IMAGE, 0)
    mask = cv2.imread(MASK_IMAGE, 0)
    if im_ref is None:
        raise FileNotFoundError(f"Reference image not found: {REF_IMAGE}")
    if mask is None:
        raise FileNotFoundError(f"Mask image not found: {MASK_IMAGE}")

    images = sorted(glob(os.path.join(TRACTION_DIR, "*")))
    if not images:
        raise FileNotFoundError(f"No images found in: {TRACTION_DIR}")

    r0, r1, c0, c1 = ROI
    im_ref_masked = im_ref * mask

    for idx, path in enumerate(images):
        print(f"[{idx + 1}/{len(images)}] {path}")
        im_def = cv2.imread(path, 0)
        if im_def is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        u, v = dic_displacement(
            im_ref_masked,
            im_def * mask,
            alpha=ALPHA,
            delta=DELTA,
            gamma=GAMMA,
            epsilon=EPSILON,
            iterations=ITERATIONS,
        )

        np.save(os.path.join(OUT_DIR_U, f"U_{idx + 1}.npy"), u[r0:r1, c0:c1])
        np.save(os.path.join(OUT_DIR_V, f"V_{idx + 1}.npy"), v[r0:r1, c0:c1])

    print(f"Done. Saved {len(images)} U/V field pairs.")


if __name__ == "__main__":
    main()
