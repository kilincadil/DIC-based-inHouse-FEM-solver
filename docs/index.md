# DIC-based in-house FEM

Turning measured DIC kinematics into a mechanically admissible stress and
strain field.

Digital image correlation gives displacements on the surface of a specimen.
Differentiating them yields a strain estimate, but one that satisfies neither
equilibrium nor any constitutive law, and that amplifies measurement noise. The
code documented here instead poses a boundary-value problem: measured
displacements are prescribed on the boundary, and the interior is solved so that
compatibility, the material law and force balance all hold simultaneously.

The same problem can be solved two ways, which is the point of the repository:

- **In Abaqus**, by generating an input file in which every pixel becomes an
  element carrying its own measured material properties.
- **In the in-house solver**, which takes the identical inputs directly in
  Python and needs no external solver or licence.

The second was written against the first, so the Abaqus path is kept as the
reference oracle rather than as a historical artefact.

```{toctree}
:maxdepth: 2
:caption: Contents

solver
abaqus
legacy
```

## Repository layout

| Folder | Contents |
|---|---|
| `solver/` | The in-house finite-element solver and its tiling diagnostic. |
| `abaqus/` | Current Abaqus pipeline: `.inp` generation and ODB extraction. |
| `legacy/` | The original DIC-to-Abaqus mesh generators, kept as first published. |
| `docs/` | These pages. |

## The model

Both paths solve the same problem:

- **Element** — CPS4, plane stress, 2x2 Gauss quadrature, one element per pixel
- **Kinematics** — small strain
- **Elasticity** — isotropic, $E = 205\,000$ MPa, $\nu = 0.3$
- **Plasticity** — associative von Mises with Ludwik hardening,

  $$\sigma_y(\varepsilon_p) = \sigma_{y0} + K\,\varepsilon_p^{\,n}$$

- **Heterogeneity** — $\sigma_{y0}$ and $K$ vary from element to element, so a
  measured per-pixel material map is carried directly into the model
- **Boundary conditions** — measured displacements prescribed at every node of
  all four edges; the interior is unknown and solved for

## What the results are, and are not

The reconstructed field belongs to an experiment that has already happened. The
material maps are derived from that same experiment, so they are *effective
descriptors* rather than independently identified grain properties. This makes
the result a reconstruction, not a forward prediction: predicting a new
experiment would require descriptors obtained independently of it.

Worth keeping separate when comparing against DIC: total strain has a direct
experimental counterpart, whereas PEEQ is a path-dependent internal variable of
the model with no measured equivalent.

## Installation

```bash
pip install -r requirements.txt
```

NumPy and SciPy are the only requirements. `pypardiso` is optional and gives a
large speed-up on bigger meshes. The Abaqus scripts need an Abaqus/Standard
installation; the in-house solver does not.

## Licence

GNU Lesser General Public License v2.1 — see `LICENSE`.
