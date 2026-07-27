# -*- coding: utf-8 -*-
"""
Abaqus/Standard input-file (.inp) writer for pixel-wise heterogeneous plasticity.

Writes a structured CPS4 mesh in which **every element carries its own material
definition**, so a per-pixel yield strength / hardening map measured from an
experiment can be transferred one-to-one onto the finite-element model. Nodal
displacements measured by DIC are prescribed on all four edges.

The file is written in two passes:

    MeshCreator      -> *Heading, *Part, *Node, *Element, *Elset/*Solid Section,
                        *Assembly, *Instance, *Nset  (truncates/creates the file)
    MaterialInserter -> ** MATERIALS, *Step, *Boundary, *Output  (appends)

`MeshCreator.create_mesh_to_file()` must therefore run before
`MaterialInserter.insert_data()` on the same path.

Index convention
----------------
All 2-D arrays are indexed ``[i, j]`` with ``i`` along x (first axis) and ``j``
along y (second axis), matching Abaqus' own X/Y ordering. Node and element
numbering is generated with ``order='F'`` so that the flat Abaqus label and the
``[i, j]`` position stay consistent. Displacement arrays are therefore shaped
``(nx_nodes, ny_nodes)`` and element-wise material arrays ``(nx_elems, ny_elems)``
-- *not* the other way round.

Units
-----
Lengths in millimetres, stresses in MPa.

Part of: DIC-based in-house FEM (https://github.com/kilincadil/DIC-based-inHouse-FEM)
"""

import numpy as np

# Material constants shared by every element. Only the yield stress and the
# hardening coefficient vary spatially; the elastic response and the plastic
# strain grid are common to the whole model.
YOUNGS_MODULUS = 205000.0   # MPa
POISSON_RATIO = 0.3
PLASTIC_STRAIN_MAX = 0.2    # upper bound of the tabulated *Plastic curve
PLASTIC_TABLE_POINTS = 50   # rows per *Plastic table

# Node-set names written for the four edges of the part.
EDGE_NSET_NAMES = ("NODEBOTTOM", "NODETOP", "NODELEFT", "NODERIGHT")

# Abaqus allows at most 16 comma-separated entries per *Nset line.
NSET_ENTRIES_PER_LINE = 16


class MeshCreator:
    """Write the mesh, sections and assembly blocks of the input file.

    One ``*Elset`` / ``*Solid Section`` pair is emitted per element so that each
    element can later be bound to its own ``*Material``. This makes the file
    large (roughly three lines per element) but is what allows a genuinely
    per-pixel material distribution.

    Parameters
    ----------
    file_path : str
        Destination ``.inp`` path. Overwritten if it already exists.
    x_max, y_max : float
        Domain size along x and y, in the same units as ``element_size``.
    element_size : float
        Edge length of one square element.
    scale_factor : float, optional
        Multiplier applied to the node coordinates, used to convert a pixel
        grid to physical units (default 1.84, i.e. 1.84 um/pixel expressed in
        the working length unit).
    """

    def __init__(self, file_path, x_max, y_max, element_size, scale_factor=1.84):
        self.file_path = file_path
        self.nx_elems = int(round(x_max / element_size))
        self.ny_elems = int(round(y_max / element_size))
        self.nx_nodes = self.nx_elems + 1
        self.ny_nodes = self.ny_elems + 1
        self.element_size = element_size
        self.scale_factor = scale_factor

    def create_mesh_to_file(self):
        """Write nodes, elements, per-element sections, assembly and edge sets."""
        x = np.linspace(0, self.nx_elems * self.element_size, self.nx_nodes) * self.scale_factor
        y = np.linspace(0, self.ny_elems * self.element_size, self.ny_nodes) * self.scale_factor

        # order='F' keeps label <-> [i, j] consistent with Abaqus X/Y ordering.
        node_ids = np.arange(1, self.nx_nodes * self.ny_nodes + 1).reshape(
            (self.nx_nodes, self.ny_nodes), order='F')

        node_lines = []
        for i in range(self.nx_nodes):
            for j in range(self.ny_nodes):
                node_lines.append("{0}, {1:.6f}, {2:.6f}\n".format(node_ids[i, j], x[i], y[j]))

        element_ids = np.arange(1, self.nx_elems * self.ny_elems + 1).reshape(
            (self.nx_elems, self.ny_elems), order='F')

        element_lines = []
        for i in range(self.nx_elems):
            for j in range(self.ny_elems):
                # Counter-clockwise connectivity, starting at the lower-left node.
                n1 = node_ids[i, j]
                n2 = node_ids[i + 1, j]
                n3 = node_ids[i + 1, j + 1]
                n4 = node_ids[i, j + 1]
                element_lines.append("{0}, {1}, {2}, {3}, {4}\n".format(
                    element_ids[i, j], n1, n2, n3, n4))

        node_sets = []
        for name, ids in zip(EDGE_NSET_NAMES,
                             (node_ids[:, 0], node_ids[:, -1],
                              node_ids[0, :], node_ids[-1, :])):
            node_sets.extend(_format_nset(name, ids))

        with open(self.file_path, 'w') as f:
            f.write("*Heading\n** PARTS\n*Part, name=PART-1\n*NODE\n")
            f.writelines(node_lines)
            f.write("*Element, type=CPS4\n")
            f.writelines(element_lines)

            for i in range(self.nx_elems):
                for j in range(self.ny_elems):
                    eid = element_ids[i, j]
                    f.write("*Elset, elset=Set-{0}\n{0}\n".format(eid))
                    f.write("*Solid Section, elset=Set-{0}, material=Material-{0}\n".format(eid))

            f.write("*End Part\n** ASSEMBLY\n*Assembly, name=Assembly\n")
            f.write("*Instance, name=PART-1-1, part=PART-1\n*End Instance\n")
            f.writelines(node_sets)
            f.write("*End Assembly\n")


class MaterialInserter:
    """Append the per-element materials, the step, the boundary conditions and
    the output requests to an input file already containing a mesh.

    Parameters
    ----------
    file_path : str
        Path written by :class:`MeshCreator`. Opened in append mode.
    disp_x_nodes, disp_y_nodes : ndarray, shape (nx_nodes, ny_nodes)
        Prescribed nodal displacements. Only the four edges are actually
        imposed; interior values are ignored and are solved for by Abaqus.
    stress_field_elems : ndarray, shape (n_table, nx_elems, ny_elems)
        Tabulated flow stress per element, one row per plastic-strain point.
        Built by :func:`field_utils.hardening_table`.
    """

    def __init__(self, file_path, disp_x_nodes, disp_y_nodes, stress_field_elems):
        self.file_path = file_path
        self.disp_x = disp_x_nodes
        self.disp_y = disp_y_nodes
        self.stress_field = stress_field_elems

    def insert_data(self):
        """Append materials, step definition, boundary conditions and outputs."""
        n_table, nx_elems, ny_elems = self.stress_field.shape
        nx_nodes, ny_nodes = self.disp_x.shape

        if (nx_nodes, ny_nodes) != (nx_elems + 1, ny_elems + 1):
            raise ValueError(
                "displacement grid {0} is inconsistent with the element grid {1}; "
                "expected {2}. Check the [x, y] index order of the inputs.".format(
                    (nx_nodes, ny_nodes), (nx_elems, ny_elems),
                    (nx_elems + 1, ny_elems + 1)))

        plastic_strain = np.linspace(0.0, PLASTIC_STRAIN_MAX, n_table)
        lines = ["** MATERIALS\n"]

        element_ids = np.arange(1, nx_elems * ny_elems + 1).reshape(
            (nx_elems, ny_elems), order='F')

        for i in range(nx_elems):
            for j in range(ny_elems):
                eid = element_ids[i, j]
                stress_vals = self.stress_field[:, i, j]
                lines.append("*Material, name=Material-{0}\n".format(eid))
                lines.append("*Elastic\n{0}., {1}\n*Plastic\n".format(
                    YOUNGS_MODULUS, POISSON_RATIO))
                lines.append("\n".join(
                    "{0:.6f}, {1:.6f}".format(stress_vals[k], plastic_strain[k])
                    for k in range(n_table)) + "\n")

        lines.append("** STEP: Step-1\n*Step, name=Step-1, nlgeom=NO, inc=1000\n")
        lines.append("*Static\n0.001, 1., 1e-06, 1.\n** BOUNDARY CONDITIONS\n")

        node_ids = np.arange(1, nx_nodes * ny_nodes + 1).reshape(
            (nx_nodes, ny_nodes), order='F')

        # Bottom and top edges span the full width; left and right edges skip the
        # corners, which are already constrained above, to avoid duplicates.
        for i in range(nx_nodes):
            lines.append(_format_bc("Bottom", node_ids[i, 0],
                                    self.disp_x[i, 0], self.disp_y[i, 0]))
            lines.append(_format_bc("Top", node_ids[i, -1],
                                    self.disp_x[i, -1], self.disp_y[i, -1]))

        for j in range(1, ny_nodes - 1):
            lines.append(_format_bc("Left", node_ids[0, j],
                                    self.disp_x[0, j], self.disp_y[0, j]))
            lines.append(_format_bc("Right", node_ids[-1, j],
                                    self.disp_x[-1, j], self.disp_y[-1, j]))

        lines.append("*Restart, write, frequency=0\n*Output, field\n*Node Output\nRF, U\n")
        lines.append("*Element Output, directions=YES\nE, PE, PEEQ, PEMAG, S\n")
        lines.append("*Output, history, variable=PRESELECT\n*End Step\n")

        with open(self.file_path, 'a') as f:
            f.writelines(lines)


def _format_nset(name, ids):
    """Return the lines of one ``*Nset`` block, wrapped to Abaqus' line limit."""
    lines = ["*Nset, nset={0}, instance=PART-1-1\n".format(name)]
    for k in range(0, len(ids), NSET_ENTRIES_PER_LINE):
        lines.append(", ".join(str(v) for v in ids[k:k + NSET_ENTRIES_PER_LINE]) + "\n")
    return lines


def _format_bc(name, node_id, u1, u2):
    """Return one ``*Boundary`` block prescribing both in-plane components."""
    return ("** Name: {0}_{1}\n*Boundary\n"
            "PART-1-1.{1}, 1, 1, {2}\n"
            "PART-1-1.{1}, 2, 2, {3}\n".format(name, node_id, u1, u2))
