# DIC-based in-house FEM

Turning measured DIC kinematics into a mechanically admissible stress and strain
field.

Digital image correlation gives displacements on the surface of a specimen.
Differentiating them yields a strain estimate — but one that satisfies neither
equilibrium nor any constitutive law, and that amplifies measurement noise. This
repository instead poses a boundary-value problem: measured displacements are
prescribed on the boundary, and the interior is solved so that compatibility,
the material law and force balance all hold at once.

The result is not a smoothed measurement. It is a field constrained
simultaneously by kinematics, material behaviour and equilibrium.

The same problem can be solved two ways, which is the point of the repository:

- **In Abaqus**, by generating an input file where every pixel becomes an
  element carrying its own measured material properties.
- **In the in-house solver**, which takes identical inputs directly in Python
  and needs no external solver and no licence.

The second was written against the first, so the Abaqus path is kept as the
reference oracle rather than as a historical artefact.

## Layout

| Folder | Contents |
|---|---|
| `solver/` | The in-house FEM solver (`fem_pixel.py`) and its tiling diagnostic (`seam_metrics.py`). |
| `abaqus/` | Current Abaqus pipeline: `.inp` generation and ODB extraction back onto a grid. |
| `legacy/` | The original DIC-to-Abaqus mesh generators, micro- and mesoscale, kept as first published. |
| `docs/` | Full documentation, built with Sphinx. |

## The model

Both paths solve the same problem:

- **Element** — CPS4, plane stress, 2x2 Gauss quadrature, one element per pixel
- **Kinematics** — small strain
- **Elasticity** — isotropic, E = 205 000 MPa, nu = 0.3
- **Plasticity** — associative von Mises with Ludwik hardening,
  `sy(ep) = sy0 + K * ep^n`
- **Heterogeneity** — `sy0` and `K` vary element by element, so a measured
  per-pixel material map is carried directly into the model
- **Boundary conditions** — measured displacements prescribed at every node of
  all four edges; the interior is unknown and solved for

The in-house solver uses an analytical consistent tangent — closed-form return
mapping plus the consistent tangent operator for plane-stress von Mises — with
Newton-Raphson, automatic increment cutback, and a factorized elastic stiffness
reused for the elastic predictor.

## Install

```bash
pip install -r requirements.txt
```

NumPy and SciPy are the only requirements. `pypardiso` is optional and gives a
large speed-up on bigger meshes. The scripts in `abaqus/` additionally need an
Abaqus/Standard installation; the solver does not.

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

Run the solver directly for a self-contained correctness check against a
closed-form equal-biaxial solution:

```bash
cd solver && python fem_pixel.py
```

The scripts are plain modules rather than an installed package, so run them from
inside their folder or add it to `sys.path`.

## Scope of the results

The reconstructed field belongs to an experiment that has already happened. The
material maps are derived from that same experiment, so they are *effective
descriptors* rather than independently identified grain properties — which makes
this a reconstruction, not a forward prediction.

When comparing against DIC, keep total strain and PEEQ separate: the former has
a direct experimental counterpart, the latter is a path-dependent internal
variable of the model with no measured equivalent.

## Documentation

Full documentation is in `docs/`, built with Sphinx and published via Read the
Docs. To build it locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Licence

GNU Lesser General Public License v2.1 — see [`LICENSE`](LICENSE).

## References

Csáti, Z. et al. (2021). *CRISTALX*.

Hu, Q. et al. (2025).
