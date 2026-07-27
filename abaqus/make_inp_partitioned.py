# -*- coding: utf-8 -*-
"""
Split a full DIC field into tiles and write one Abaqus .inp per tile.

A field of several thousand elements per side is impractical as a single
Abaqus job. This script cuts it into a regular grid of non-overlapping
partitions and writes an independent input file for each, every tile taking its
own boundary conditions from the measured displacement field. The solved tiles
are stitched back together afterwards.

Tiles here are hard-cut: neighbouring partitions share only their common edge.
Overlap ("padding") between tiles reduces the artefact left at the interfaces;
``seam_metrics.py`` in the repository root is the diagnostic used to calibrate
how much overlap is required.

Run it either way:

    python make_inp_partitioned.py                  # uses the CONFIG block below
    python make_inp_partitioned.py --x-size 0.9 --y-size 0.775

Every CONFIG entry has a matching ``--flag``; anything not passed falls back to
the value set here, so the file stays runnable as-is from an IDE.

Part of: DIC-based in-house FEM (https://github.com/kilincadil/DIC-based-inHouse-FEM)
"""

import argparse
import os

import numpy as np

from field_utils import (element_count_label, fill_nan_iter_4nbr,
                         hardening_table, pad_last_row_col)
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

    # Per-element material maps, as .npy arrays shaped [x, y]. NaNs are filled.
    "yield_file": "data/yield_stress.npy",
    "hardening_file": "data/hardening_coeff.npy",

    # Multiplier applied to the raw hardening map to reach MPa.
    "hardening_scale": 396.0,

    # Pixel displacements -> model length units (mm).
    "disp_scale": 0.00184,

    # Size of one tile, in mm. Choose divisors of the full field so that no
    # remainder strip is left over: the tiling below drops any partial tile.
    "x_size": 0.72,
    "y_size": 0.62,
    "element_size": 0.001,

    # Node-coordinate multiplier, i.e. um per pixel.
    "scale_factor": 1.84,

    # Ludwik hardening exponent.
    "n_exp": 0.245,

    "out_dir": "inp_partitions",
    "name": "mesh_part",
}


def build_partitions(cfg):
    """Write one .inp per tile. Returns the list of paths written."""
    nx_elems = int(round(cfg["x_size"] / cfg["element_size"]))
    ny_elems = int(round(cfg["y_size"] / cfg["element_size"]))

    disp_x = np.load(cfg["disp_x_file"]) * cfg["disp_scale"]
    disp_y = np.load(cfg["disp_y_file"]) * cfg["disp_scale"]

    # Displacements are supplied on the element grid but prescribed at nodes,
    # so extend by one row and column before slicing tiles out of them.
    disp_x = pad_last_row_col(disp_x)
    disp_y = pad_last_row_col(disp_y)

    yield_map = fill_nan_iter_4nbr(np.load(cfg["yield_file"]))
    K_map = fill_nan_iter_4nbr(np.load(cfg["hardening_file"]) * cfg["hardening_scale"])

    stresses = hardening_table(yield_map, K_map, cfg["n_exp"])

    n_tiles_x = (disp_x.shape[0] - 1) // nx_elems
    n_tiles_y = (disp_x.shape[1] - 1) // ny_elems

    if n_tiles_x < 1 or n_tiles_y < 1:
        raise ValueError(
            "tile of {0}x{1} elements does not fit in a field of {2}".format(
                nx_elems, ny_elems, (disp_x.shape[0] - 1, disp_x.shape[1] - 1)))

    remainder_x = (disp_x.shape[0] - 1) - n_tiles_x * nx_elems
    remainder_y = (disp_x.shape[1] - 1) - n_tiles_y * ny_elems
    if remainder_x or remainder_y:
        print("WARNING: {0}x{1} elements at the far edges are not covered by any "
              "tile. Pick a tile size that divides the field exactly to avoid "
              "discarding them.".format(remainder_x, remainder_y))

    out_dir = cfg["out_dir"]
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    label = element_count_label(nx_elems, ny_elems)
    expected_disp = (nx_elems + 1, ny_elems + 1)
    expected_stress = (stresses.shape[0], nx_elems, ny_elems)

    written = []
    partition_id = 0

    for i in range(n_tiles_x):
        for j in range(n_tiles_y):
            i0 = i * nx_elems
            j0 = j * ny_elems

            disp_x_tile = disp_x[i0:i0 + nx_elems + 1, j0:j0 + ny_elems + 1]
            disp_y_tile = disp_y[i0:i0 + nx_elems + 1, j0:j0 + ny_elems + 1]
            stress_tile = stresses[:, i0:i0 + nx_elems, j0:j0 + ny_elems]

            for name, got, want in (("disp_x", disp_x_tile.shape, expected_disp),
                                    ("disp_y", disp_y_tile.shape, expected_disp),
                                    ("stress", stress_tile.shape, expected_stress)):
                if got != want:
                    raise RuntimeError(
                        "bad {0} tile shape at partition {1}: got {2}, expected {3}".format(
                            name, partition_id, got, want))

            file_path = os.path.join(out_dir, "{0}_{1}_{2}.inp".format(
                cfg["name"], partition_id, label))

            mesh = MeshCreator(file_path, cfg["x_size"], cfg["y_size"],
                               cfg["element_size"], cfg["scale_factor"])
            mesh.create_mesh_to_file()

            materials = MaterialInserter(file_path, disp_x_tile, disp_y_tile, stress_tile)
            materials.insert_data()

            print("partition {0}: tile ({1}/{2}, {3}/{4})".format(
                partition_id, i + 1, n_tiles_x, j + 1, n_tiles_y))

            written.append(file_path)
            partition_id += 1

    return written


def parse_args():
    """Overlay command-line flags onto CONFIG and return the merged dict."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])

    for key, default in CONFIG.items():
        parser.add_argument("--" + key.replace("_", "-"), dest=key,
                            type=type(default), default=default)

    return vars(parser.parse_args())


if __name__ == "__main__":
    config = parse_args()
    paths = build_partitions(config)
    print("\n{0} partitions written to: {1}".format(len(paths), config["out_dir"]))
