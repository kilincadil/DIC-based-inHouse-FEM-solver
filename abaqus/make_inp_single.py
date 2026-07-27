# -*- coding: utf-8 -*-
"""
Generate one Abaqus .inp file for a centred crop of a DIC field.

Reads measured displacement fields and per-pixel material maps, crops the
central region to the requested size, and writes a single input file in which
every element carries its own tabulated flow curve.

Use this for a single window (validation, calibration, a reference solve). For
a full field split into several jobs, use ``make_inp_partitioned.py`` instead.

Run it either way:

    python make_inp_single.py                       # uses the CONFIG block below
    python make_inp_single.py --x-size 0.5 --y-size 0.5 --out-dir ./inp

Every CONFIG entry has a matching ``--flag``; anything not passed falls back to
the value set here, so the file stays runnable as-is from an IDE.

Part of: DIC-based in-house FEM (https://github.com/kilincadil/DIC-based-inHouse-FEM)
"""

import argparse
import os

import numpy as np

from field_utils import crop_center, element_count_label, hardening_table
from inp_writer import MaterialInserter, MeshCreator

# --------------------------------------------------------------------------- #
# CONFIG - edit these, or override any of them from the command line.
# --------------------------------------------------------------------------- #
CONFIG = {
    # Measured displacement fields, as .npy arrays shaped [x, y] in pixels.
    # NOTE: in the reference dataset the DIC "U" channel is the y displacement
    # and "V" is the x displacement; adjust these two paths to your convention.
    "disp_x_file": "data/V.npy",
    "disp_y_file": "data/U.npy",

    # Per-element material maps, as .npy arrays shaped [x, y].
    "yield_file": "data/yield_stress.npy",
    "hardening_file": "data/hardening_coeff.npy",

    # Multiplier applied to the raw hardening map to reach MPa.
    "hardening_scale": 396.0,

    # Pixel displacements -> model length units (mm).
    "disp_scale": 0.00184,

    # Domain and discretisation, in mm. One element per pixel.
    "x_size": 0.1,
    "y_size": 0.1,
    "element_size": 0.001,

    # Node-coordinate multiplier, i.e. um per pixel.
    "scale_factor": 1.84,

    # Ludwik hardening exponent.
    "n_exp": 0.245,

    "out_dir": "inp",
    "name": "mesh_central",
}


def build_inp(cfg):
    """Write one .inp for the centred crop described by ``cfg``. Returns its path."""
    nx_elems = int(round(cfg["x_size"] / cfg["element_size"]))
    ny_elems = int(round(cfg["y_size"] / cfg["element_size"]))

    disp_x = np.load(cfg["disp_x_file"]) * cfg["disp_scale"]
    disp_y = np.load(cfg["disp_y_file"]) * cfg["disp_scale"]

    yield_map = np.load(cfg["yield_file"])
    K_map = np.load(cfg["hardening_file"]) * cfg["hardening_scale"]

    # Displacements live on nodes (one more than elements per axis), material
    # maps on elements. Both are cropped [x, y] about the centre of the field.
    disp_x = crop_center(disp_x, nx_elems + 1, ny_elems + 1)
    disp_y = crop_center(disp_y, nx_elems + 1, ny_elems + 1)
    yield_map = crop_center(yield_map, nx_elems, ny_elems)
    K_map = crop_center(K_map, nx_elems, ny_elems)

    stresses = hardening_table(yield_map, K_map, cfg["n_exp"])

    out_dir = cfg["out_dir"]
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    file_path = os.path.join(out_dir, "{0}_{1}.inp".format(
        cfg["name"], element_count_label(nx_elems, ny_elems)))

    mesh = MeshCreator(file_path, cfg["x_size"], cfg["y_size"],
                       cfg["element_size"], cfg["scale_factor"])
    mesh.create_mesh_to_file()

    materials = MaterialInserter(file_path, disp_x, disp_y, stresses)
    materials.insert_data()

    return file_path


def parse_args():
    """Overlay command-line flags onto CONFIG and return the merged dict."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])

    for key, default in CONFIG.items():
        parser.add_argument("--" + key.replace("_", "-"), dest=key,
                            type=type(default), default=default)

    return vars(parser.parse_args())


if __name__ == "__main__":
    path = build_inp(parse_args())
    print("Input file written: {0}".format(path))
