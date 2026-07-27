# -*- coding: utf-8 -*-
"""
Export an Abaqus ODB onto a regular grid of NumPy arrays.

PYTHON 2.7 -- this module runs inside Abaqus' own interpreter, not in your
Python 3 environment. Launch it with ``abaqus python odb_to_grid.py ...`` or,
more conveniently, through ``run_odb_export.py``.

The mesh written by ``inp_writer.py`` is structured and axis-aligned, so every
node sits at an integer multiple of the pixel pitch. Node and element positions
are therefore recovered from their coordinates rather than from their labels,
which Abaqus does not guarantee to be contiguous or ordered. Elements are
placed by their lower-left node, and integration-point values are averaged over
each element.

Outputs, all shaped ``[frame, x, y]`` and named after the ODB:

    <base>_u1.npy, <base>_u2.npy        nodal displacement    (n_frames, nx_nodes, ny_nodes)
    <base>_rf1.npy, <base>_rf2.npy      nodal reaction force  (n_frames, nx_nodes, ny_nodes)
    <base>_rf_total.npy                 summed reaction force (n_frames, 4, 2)
    <base>_s11/_s22/_s12.npy            element stress        (n_frames, nx_elems, ny_elems)
    <base>_e11/_e22/_e12.npy            element total strain  (n_frames, nx_elems, ny_elems)
    <base>_peeq.npy                     equivalent plastic strain (n_frames, nx_elems, ny_elems)

``rf_total`` holds the summed RF1 and RF2 over the four prescribed edges, in
the order bottom, top, left, right.

Part of: DIC-based in-house FEM (https://github.com/kilincadil/DIC-based-inHouse-FEM)
"""

import os
import sys

import numpy as np
from abaqusConstants import INTEGRATION_POINT, NODAL
from odbAccess import openOdb

# Edge node sets written by inp_writer.MeshCreator, in reporting order.
EDGE_NSETS = ('NODEBOTTOM', 'NODETOP', 'NODELEFT', 'NODERIGHT')


def process_odb_griddata(odb_path, output_dir, step_name='Step-1',
                         instance_name='PART-1-1', m_per_pixel=0.00184):
    """Read one ODB and write its fields as gridded .npy arrays.

    Parameters
    ----------
    odb_path : str
        Path to the .odb file.
    output_dir : str
        Directory for the .npy outputs. Created if missing.
    step_name, instance_name : str
        Step and part-instance to read.
    m_per_pixel : float
        Grid pitch used to convert node coordinates to integer indices. Must
        match the ``scale_factor`` times ``element_size`` used when the mesh was
        written, otherwise the rounding below will collapse distinct nodes.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    odb = openOdb(path=odb_path, readOnly=True)

    try:
        step = odb.steps[step_name]
        inst = odb.rootAssembly.instances[instance_name]

        n_nodes = len(inst.nodes)
        n_elems = len(inst.elements)

        node_index = dict((n.label, i) for i, n in enumerate(inst.nodes))
        elem_index = dict((e.label, i) for i, e in enumerate(inst.elements))
        node_by_label = dict((n.label, n) for n in inst.nodes)

        # --- map nodes onto the structured grid ----------------------------- #
        coords = np.array([n.coordinates[:2] for n in inst.nodes], dtype=float)
        ix = np.rint(coords[:, 0] / m_per_pixel).astype(int)
        iy = np.rint(coords[:, 1] / m_per_pixel).astype(int)

        x_unique = np.unique(ix)
        y_unique = np.unique(iy)
        nx_nodes = len(x_unique)
        ny_nodes = len(y_unique)
        nx_elems = nx_nodes - 1
        ny_elems = ny_nodes - 1

        if nx_nodes * ny_nodes != n_nodes:
            raise RuntimeError(
                "grid %dx%d does not account for all %d nodes; check m_per_pixel"
                % (nx_nodes, ny_nodes, n_nodes))

        x_to_j = dict((x, j) for j, x in enumerate(x_unique))
        y_to_j = dict((y, j) for j, y in enumerate(y_unique))
        node_jx = np.array([x_to_j[int(v)] for v in ix], dtype=int)
        node_jy = np.array([y_to_j[int(v)] for v in iy], dtype=int)

        # --- map elements by their lower-left node -------------------------- #
        elem_ll_ix = np.zeros(n_elems, dtype=int)
        elem_ll_iy = np.zeros(n_elems, dtype=int)

        for e in inst.elements:
            loc = elem_index[e.label]
            conn_ix = []
            conn_iy = []
            for nl in e.connectivity:
                node = node_by_label[nl]
                conn_ix.append(int(round(node.coordinates[0] / m_per_pixel)))
                conn_iy.append(int(round(node.coordinates[1] / m_per_pixel)))
            elem_ll_ix[loc] = min(conn_ix)
            elem_ll_iy[loc] = min(conn_iy)

        x0_to_j = dict((x, j) for j, x in enumerate(np.sort(np.unique(elem_ll_ix))))
        y0_to_j = dict((y, j) for j, y in enumerate(np.sort(np.unique(elem_ll_iy))))
        elem_jx = np.array([x0_to_j[int(v)] for v in elem_ll_ix], dtype=int)
        elem_jy = np.array([y0_to_j[int(v)] for v in elem_ll_iy], dtype=int)

        # --- edge node sets, for the reaction-force totals ------------------ #
        edge_members = []
        for name in EDGE_NSETS:
            if name in odb.rootAssembly.nodeSets:
                labels = set(n.label for n in odb.rootAssembly.nodeSets[name].nodes[0])
                edge_members.append(labels)
            else:
                print("WARNING: node set %s absent from the ODB; its reaction "
                      "total will be reported as zero." % name)
                edge_members.append(set())

        # --- per-frame extraction ------------------------------------------- #
        frames = {'u1': [], 'u2': [], 'rf1': [], 'rf2': [], 'rf_total': [],
                  's11': [], 's22': [], 's12': [],
                  'e11': [], 'e22': [], 'e12': [], 'peeq': []}

        for fr in step.frames:
            u1, u2 = _nodal_pair(fr, 'U', inst, node_index, n_nodes)
            rf1, rf2 = _nodal_pair(fr, 'RF', inst, node_index, n_nodes)

            for key, vals in (('u1', u1), ('u2', u2), ('rf1', rf1), ('rf2', rf2)):
                frames[key].append(_to_grid(vals, node_jx, node_jy, nx_nodes, ny_nodes))

            totals = np.zeros((len(EDGE_NSETS), 2), dtype=float)
            for k, labels in enumerate(edge_members):
                for label in labels:
                    if label in node_index:
                        totals[k, 0] += rf1[node_index[label]]
                        totals[k, 1] += rf2[node_index[label]]
            frames['rf_total'].append(totals)

            s = _element_components(fr, 'S', inst, elem_index, n_elems, (0, 1, 3))
            e = _element_components(fr, 'E', inst, elem_index, n_elems, (0, 1, 3))
            peeq = _element_components(fr, 'PEEQ', inst, elem_index, n_elems, (0,))

            for key, vals in (('s11', s[0]), ('s22', s[1]), ('s12', s[2]),
                              ('e11', e[0]), ('e22', e[1]), ('e12', e[2]),
                              ('peeq', peeq[0])):
                frames[key].append(_to_grid(vals, elem_jx, elem_jy, nx_elems, ny_elems))

    finally:
        odb.close()

    base = os.path.splitext(os.path.basename(odb_path))[0]
    for key, stack in frames.items():
        np.save(os.path.join(output_dir, '%s_%s.npy' % (base, key)), np.array(stack))

    print("Exported %d frames on a %dx%d node grid for %s"
          % (len(step.frames), nx_nodes, ny_nodes, base))


def _nodal_pair(frame, field_name, instance, node_index, n_nodes):
    """Return the two in-plane components of a nodal field, indexed by node."""
    comp1 = np.zeros(n_nodes, dtype=float)
    comp2 = np.zeros(n_nodes, dtype=float)

    if field_name not in frame.fieldOutputs:
        return comp1, comp2

    subset = frame.fieldOutputs[field_name].getSubset(region=instance, position=NODAL)
    for v in subset.values:
        i = node_index[v.nodeLabel]
        comp1[i] = v.data[0]
        comp2[i] = v.data[1]

    return comp1, comp2


def _element_components(frame, field_name, instance, elem_index, n_elems, components):
    """Return element-averaged integration-point values for the given components.

    Abaqus reports one value per integration point; CPS4 with 2x2 quadrature
    gives four per element, which are averaged here to a single element value.
    """
    out = [np.zeros(n_elems, dtype=float) for _ in components]

    if field_name not in frame.fieldOutputs:
        return out

    subset = frame.fieldOutputs[field_name].getSubset(region=instance,
                                                      position=INTEGRATION_POINT)
    counts = np.zeros(n_elems, dtype=int)

    for v in subset.values:
        loc = elem_index[v.elementLabel]
        # Scalar fields such as PEEQ arrive as a bare float, not a sequence.
        data = v.data if hasattr(v.data, '__len__') else (v.data,)
        for k, c in enumerate(components):
            out[k][loc] += data[c]
        counts[loc] += 1

    nonzero = counts > 0
    for arr in out:
        arr[nonzero] /= counts[nonzero]

    return out


def _to_grid(values, jx, jy, nx, ny):
    """Scatter a flat per-entity array onto an ``[x, y]`` grid, NaN where absent."""
    grid = np.empty((nx, ny), dtype=float)
    grid[:] = np.nan
    grid[jx, jy] = values
    return grid


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit("usage: abaqus python odb_to_grid.py <odb> <out_dir> "
                 "[step] [instance] [m_per_pixel]")

    process_odb_griddata(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3] if len(sys.argv) > 3 else 'Step-1',
        sys.argv[4] if len(sys.argv) > 4 else 'PART-1-1',
        float(sys.argv[5]) if len(sys.argv) > 5 else 0.00184,
    )
