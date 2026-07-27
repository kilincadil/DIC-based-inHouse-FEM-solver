# In-house solver

`solver/fem_pixel.py` is a pure-Python finite-element solver written as a
drop-in replacement for Abaqus in DIC-driven, pixel-wise heterogeneous
plasticity problems. It takes the same inputs the Abaqus path would receive —
measured boundary displacements plus per-element yield and hardening maps — and
solves the same boundary-value problem in-process, with no external solver and
no licence.

## What it solves

- **Element** — CPS4 (4-node quad, plane stress), 2x2 Gauss quadrature
- **Material** — von Mises plasticity with either

  - `'ludwik'` — the analytic curve $\sigma_y = \sigma_{y0} + K\varepsilon_p^n$, or
  - `'tabular'` — a piecewise-linear table reproducing how Abaqus interpolates
    a `*Plastic` table, for direct cross-validation

- **Boundary conditions** — all four edges prescribed from measured displacement
  fields
- **Heterogeneity** — yield stress and hardening coefficient vary per element
- **Solver** — incremental loading with Newton-Raphson and an **analytical
  consistent tangent** (closed-form return mapping plus the consistent tangent
  operator for plane-stress von Mises — no numerical tangent, no external
  material subroutine), automatic increment cutback on non-convergence, and a
  factorized elastic stiffness reused for the elastic predictor

The two hardening modes exist for different jobs. `'tabular'` matches Abaqus
term for term and is what makes a like-for-like comparison meaningful;
`'ludwik'` is the analytic reference and keeps hardening beyond the point where
a finite table would plateau.

## Quick start

```python
import numpy as np
from fem_pixel import run_fem

nx, ny = 100, 100
element_size = 0.001     # mm

# boundary displacements at every node, e.g. from a DIC U/V field (mm)
disp_x = np.zeros((nx + 1, ny + 1))
disp_y = np.zeros((nx + 1, ny + 1))
# ... fill disp_x/disp_y with measured edge displacements ...

# per-element material maps, heterogeneous down to one value per pixel
yield_map = np.full((nx, ny), 250.0)   # sy0, MPa
K_map     = np.full((nx, ny), 500.0)   # hardening coefficient

result = run_fem(
    disp_x, disp_y, yield_map, K_map, n_exp=0.245,
    x_size=nx * element_size, y_size=ny * element_size,
    element_size=element_size, scale_factor=1.0,
    E_mod=205000.0, nu=0.3, N_inc=20,
    hardening='ludwik', verbose=True,
)
```

The scripts are plain modules rather than an installed package, so run them from
inside `solver/` or add that folder to `sys.path`.

### Results

| Key | Shape | Meaning |
|---|---|---|
| `result['S']` | `(nx, ny, 3)` | element stress, `[s11, s22, s12]` |
| `result['E']` | `(nx, ny, 3)` | element strain, `[e11, e22, g12]` |
| `result['PEEQ']` | `(nx, ny)` | equivalent plastic strain |
| `result['U']` | `(nx+1, ny+1, 2)` | nodal displacement |
| `result['RF']` | — | nodal reaction forces on the prescribed edges |

### Verification

Running the module directly solves an equal-biaxial tension case that has a
closed-form solution, and reports the error against it:

```bash
python fem_pixel.py
```

### Load history

To get a stress/strain-versus-load curve without re-solving, ask for
intermediate snapshots:

```python
result = run_fem(..., snapshot_fractions=[0.1, 0.2, ..., 1.0])
result['frames'][0.5]   # {'S':..., 'E':..., 'PEEQ':..., 'U':...} at 50% load
```

## Scaling to large fields

A full DIC field of several thousand elements per side is too large for one
solve. The same solver handles it tile by tile: split the field, solve each tile
independently — each taking its own edge conditions from the measured global
field — then stitch the tiles back together, either by keeping each tile's
non-overlapping core or by tent-weighted blending across the overlap.

The overlap, or *padding*, has to be wide enough that each tile's own artificial
boundary has decayed before the region you keep begins. `solver/seam_metrics.py`
measures exactly that: given a stitched element-stress field, it computes the
assembled nodal force-equilibrium residual at the tile interfaces and its decay
with distance from them, so the required padding can be calibrated from
measurement rather than guessed.

:::{note}
Calibrate padding with tiles that stay much larger than the padding being
tested. As padding approaches the tile size, each tile grows to cover most of
the domain and the comparison silently becomes a monolithic solve against
itself — the residual then converges for reasons that have nothing to do with
the decay length being measured.
:::

The domain-decomposition driver and stitching scripts are not part of this
repository; what is here is the solver and the calibration diagnostic.
