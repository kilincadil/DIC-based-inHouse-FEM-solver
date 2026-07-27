# Abaqus workflow

The reference pipeline. These scripts turn a measured DIC field into Abaqus
input files and read the results back onto a regular grid, so they can be
compared field by field with the in-house solver.

Keeping this path working matters: `fem_pixel.py` is validated *against* it, so
it is the oracle rather than a leftover.

## Pipeline

```text
DIC displacement + material maps (.npy)
                 |
                 |  make_inp_single.py        one centred window
                 |  make_inp_partitioned.py   full field, one job per tile
                 v
            Abaqus .inp
                 |
                 |  Abaqus/Standard   (run separately)
                 v
              .odb
                 |
                 |  run_odb_export.py  ->  abaqus python odb_to_grid.py
                 v
      gridded fields (.npy), shaped [frame, x, y]
```

## Files

| File | Runs under | Purpose |
|---|---|---|
| `inp_writer.py` | Python 3 | Writes the mesh, per-element materials, step and boundary conditions. |
| `field_utils.py` | Python 3 | Cropping, NaN filling, and the Ludwik hardening-table builder. |
| `make_inp_single.py` | Python 3 | One `.inp` for a centred crop of the field. |
| `make_inp_partitioned.py` | Python 3 | One `.inp` per tile, for fields too large for a single job. |
| `run_odb_export.py` | Python 3 | Launches the export below inside Abaqus. Accepts a glob for batches. |
| `odb_to_grid.py` | **Python 2.7, inside Abaqus** | Reads an ODB and writes gridded `.npy` arrays. |

Only `odb_to_grid.py` needs Abaqus' own interpreter; everything else is plain
Python 3 with NumPy.

## Model

Identical to the in-house solver: CPS4 plane stress, one element per pixel,
isotropic elasticity at $E = 205\,000$ MPa and $\nu = 0.3$, and von Mises
plasticity with a Ludwik curve tabulated at 50 points over
$\varepsilon_p \in [0, 0.2]$.

Because $\sigma_{y0}$ and $K$ vary per element, each element gets its own
`*Material` block. That is what makes the input files large — roughly three
lines per element for the sections, plus fifty more for the table.

The elastic constants and the plastic-strain grid are named constants at the top
of `inp_writer.py`: `YOUNGS_MODULUS`, `POISSON_RATIO`, `PLASTIC_STRAIN_MAX` and
`PLASTIC_TABLE_POINTS`.

## Index convention

Every 2-D array is `[i, j]` with **i along x and j along y**, matching Abaqus'
X/Y ordering. Labels are generated with `order='F'` so a flat Abaqus label and
an `[i, j]` position stay in step.

Displacement arrays are therefore `(nx_nodes, ny_nodes)` and element maps
`(nx_elems, ny_elems)`.

:::{warning}
On a square domain a transposed array is invisible — it only corrupts the model
once the window stops being square. `MaterialInserter.insert_data` raises if the
displacement and element grids disagree, which catches the common case.
:::

## Usage

Each driver carries a `CONFIG` block at the top and a matching set of
command-line flags. Edit the block and press run in an IDE, or override
individual entries from a terminal — anything not passed falls back to `CONFIG`.

```bash
# one centred 0.5 x 0.5 mm window
python make_inp_single.py --x-size 0.5 --y-size 0.5 --out-dir inp

# a full field in 0.9 x 0.775 mm tiles
python make_inp_partitioned.py --x-size 0.9 --y-size 0.775 --out-dir inp_partitions

# read the results back (add --abaqus-cmd if abaqus is not on PATH)
python run_odb_export.py --odb "runs/*.odb" --output-dir export
```

### Inputs

Four `.npy` arrays, all indexed `[x, y]` and all on the **DIC measurement
grid** — one value per pixel, that is, one per element:

| Array | Unit |
|---|---|
| displacement x, y | pixels, scaled by `disp_scale` to mm |
| yield stress $\sigma_{y0}$ | MPa |
| hardening coefficient $K$ | scaled by `hardening_scale` to MPa |

Displacements are prescribed at *nodes*, of which there is one more per axis
than there are elements. The two drivers reconcile that differently:
`make_inp_single.py` crops an `(nx+1, ny+1)` node block out of the middle, which
is why the window must be strictly smaller than the field;
`make_inp_partitioned.py` covers the whole field, so it extends the array by one
row and column instead.

:::{note}
In the reference dataset the DIC `U` channel carries the **y** displacement and
`V` the **x** displacement. The `CONFIG` defaults reflect that; check it against
your own export before trusting the result.
:::

`make_inp_partitioned.py` fills NaNs in the material maps by iterated
four-neighbour averaging, which DIC fields generally need. `make_inp_single.py`
does not, assuming a hand-picked central window is already clean — import
`fill_nan_iter_4nbr` from `field_utils` if that is not true of your crop.

### Outputs

All arrays are `[frame, x, y]`, named after the ODB:

| File | Grid |
|---|---|
| `_u1`, `_u2` | nodes |
| `_rf1`, `_rf2` | nodes |
| `_rf_total` | `(n_frames, 4, 2)` — RF1 and RF2 summed over bottom, top, left, right |
| `_s11`, `_s22`, `_s12` | elements |
| `_e11`, `_e22`, `_e12` | elements |
| `_peeq` | elements |

Integration-point values are averaged over each element's four Gauss points.
Positions are recovered from node coordinates rather than labels, since Abaqus
does not guarantee labels are contiguous or ordered.

## Tiling

`make_inp_partitioned.py` cuts hard: neighbouring tiles share only their common
edge, with no overlap. Each tile takes its boundary conditions from the measured
field, so the interfaces are consistent in displacement but still carry an
artefact in the derived quantities.

Pick a tile size that divides the field exactly — a partial tile at the far edge
is skipped, and the script warns when that happens. Reducing the interface
artefact means solving each tile on a padded region and keeping only its core;
`solver/seam_metrics.py` measures the residual at the interfaces and its decay
with distance, which is how the required padding is calibrated.
