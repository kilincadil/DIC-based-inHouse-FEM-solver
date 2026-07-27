# DIC-based in-house FEM

Finite-element reconstruction of stress and strain fields from digital image
correlation (DIC) measurements.

DIC measures displacements on a specimen surface. Differentiating those
displacements gives a strain estimate, but that estimate enforces neither
equilibrium nor a constitutive law, and differentiation amplifies measurement
noise. This repository takes a different route: the measured displacements are
imposed as boundary conditions on a finite-element model, and the interior
fields are computed from equilibrium and the material law. The output satisfies
compatibility, the constitutive relation and force balance simultaneously. It is
a mechanical solution driven by measurement, not a filtered version of the
measurement.

## Two implementations

The same boundary-value problem is solved along two independent paths.

**Abaqus.** `abaqus/` writes an input file in which each DIC pixel becomes one
element carrying its own measured material properties, and extracts the results
from the resulting ODB onto a regular grid.

**In-house solver.** `solver/fem_pixel.py` accepts the same inputs directly in
Python and solves the problem in-process, without Abaqus and without a
commercial licence.

The in-house solver was developed and validated against the Abaqus path. The
Abaqus scripts are maintained as the reference implementation against which the
solver is checked, not retained for historical reasons.

## Layout

| Folder | Contents |
|---|---|
| `solver/` | In-house FEM solver (`fem_pixel.py`) and the tiling diagnostic (`seam_metrics.py`). |
| `abaqus/` | Abaqus pipeline: `.inp` generation and ODB extraction onto a regular grid. |
| `legacy/` | Original DIC-to-Abaqus mesh generators, microscale and mesoscale, as first published. |
| `docs/` | Sphinx documentation sources. |

## Model

Both paths solve an identical problem.

| | |
|---|---|
| Element | CPS4, plane stress, 2x2 Gauss quadrature, one element per pixel |
| Kinematics | Small strain |
| Elasticity | Isotropic, E = 205 000 MPa, nu = 0.3 |
| Plasticity | Associative von Mises, Ludwik hardening `sy(ep) = sy0 + K*ep^n` |
| Heterogeneity | `sy0` and `K` vary element by element, carrying a measured per-pixel material map into the model |
| Boundary conditions | Measured displacements prescribed at every node of all four edges; interior displacements unknown |

The in-house solver integrates the constitutive law with a closed-form return
mapping and the corresponding analytical consistent tangent for plane-stress von
Mises plasticity. Global iterations use Newton-Raphson with automatic increment
cutback on non-convergence. The elastic stiffness is factorized once and reused
for each elastic predictor.

## Installation

```bash
pip install -r requirements.txt
```

NumPy and SciPy are the only requirements. `pypardiso` is optional and gives a
substantial speed-up on larger meshes. The scripts in `abaqus/` additionally
require an Abaqus/Standard installation; the in-house solver does not.

## Quick start

```python
import numpy as np
from fem_pixel import run_fem

nx, ny = 100, 100
element_size = 0.001     # mm

disp_x = np.zeros((nx + 1, ny + 1))    # measured edge displacements, mm
disp_y = np.zeros((nx + 1, ny + 1))
yield_map = np.full((nx, ny), 250.0)   # sy0, MPa, per element
K_map     = np.full((nx, ny), 500.0)   # hardening coefficient, per element

result = run_fem(
    disp_x, disp_y, yield_map, K_map, n_exp=0.245,
    x_size=nx * element_size, y_size=ny * element_size,
    element_size=element_size, scale_factor=1.0,
    E_mod=205000.0, nu=0.3, N_inc=20,
    hardening='ludwik', verbose=True,
)
# result['S'], ['E'], ['PEEQ'], ['U'], ['RF']
```

The scripts are plain modules rather than an installed package. Run them from
inside their folder, or add that folder to `sys.path`.

Executing the solver directly runs a verification case — equal-biaxial tension,
which has a closed-form solution — and reports the error against it:

```bash
cd solver && python fem_pixel.py
```

## Interpretation and limitations

The per-element maps `sy0(x, y)` and `K(x, y)` are identified from the same test
that the model reconstructs. They describe the local response effectively but
are not independently measured grain properties. Two consequences follow.

The computed fields reconstruct one specific test. Predicting a different test
would require material descriptors identified independently of it, or a
microstructure-based constitutive model with transferable parameters.

Total strain and PEEQ must be compared on different terms. Total strain has a
direct experimental counterpart in the DIC field. PEEQ is a path-dependent
internal variable of the constitutive model with no measured equivalent, and
should be read as a model diagnostic rather than as a measurement.

## Documentation

Sources are in `docs/`, built with Sphinx and published through Read the Docs.
To build locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Licence

GNU Lesser General Public License v2.1. See [`LICENSE`](LICENSE).

## References

Csáti, Z. et al. (2021). *CRISTALX*.

Hu, Q. et al. (2025).
