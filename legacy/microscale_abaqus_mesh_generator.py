"""
@author: akilinc

This script generates an Abaqus input file with mesh information and material data.
It includes classes for creating the mesh and inserting material properties with 
data mapping procedure.
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np


class MeshCreator:
    """
    A class to create a structured mesh and write it to an Abaqus input file.

    Attributes:
        file_path (str): Path to save the Abaqus input file.
        x_max (float): Maximum extent of the mesh in the x-direction.
        y_max (float): Maximum extent of the mesh in the y-direction.
        element_size (float): Size of each mesh element.
        scale_factor (float): Scale factor for node coordinates.
    """

    def __init__(self, file_path, x_max, y_max, element_size, scale_factor):
        self.file_path = file_path
        self.x_max = x_max
        self.y_max = y_max
        self.element_size = element_size
        self.scale_factor = scale_factor

    def create_mesh_nodes(self):
        """
        Creates a structured mesh grid within the specified domain.

        This method calculates the number of grid nodes along both x and y axes 
        based on the mesh size and element size, and returns the 2D coordinates 
        of the mesh grid nodes.


        Returns:
            tuple: Two 2D arrays representing the x and y coordinates of the mesh nodes.
                   - `X` (2D array): x-coordinates of the mesh grid nodes.
                   - `Y` (2D array): y-coordinates of the mesh grid nodes.
        """
        # Calculate the number of nodes for each axis
        x_size = int(self.x_max / self.element_size) + 1
        y_size = int(self.y_max / self.element_size) + 1

        # Generate linearly spaced values for x and y coordinates
        x = jnp.linspace(0, self.x_max, x_size)
        y = jnp.linspace(0, self.y_max, y_size)

        # Create a 2D meshgrid of coordinates
        X, Y = jnp.meshgrid(x, y, indexing='xy')

        # Store the mesh coordinates
        self.X = X
        self.Y = Y

        return X, Y


    @staticmethod
    @jit
    def compute_node_data(X, Y, scale_factor):
        """
        Computes node IDs and scaled coordinates.

        Args:
            X (array): X coordinates of the mesh nodes.
            Y (array): Y coordinates of the mesh nodes.
            scale_factor (float): Scale factor for node coordinates.

        Returns:
            tuple: Node IDs, scaled x coordinates, scaled y coordinates.
        """
        node_ids = jnp.arange(1, X.size + 1).reshape(X.shape, order='C')
        x_scaled = X * scale_factor
        y_scaled = Y * scale_factor
        return node_ids, x_scaled, y_scaled

    def write_node(self, node_id, prefix):
        """
        Generates a node set line for the Abaqus input file.

        Args:
            node_id (int): Node ID.
            prefix (str): Prefix for the node set name.

        Returns:
            str: Formatted node set line.
        """
        return f"*Nset, nset={prefix}{int(node_id)}, instance=PART-1-1\n{int(node_id)}\n"

    def write_element(self, element_id):
        """
        Generates an element set line for the Abaqus input file.

        Args:
            element_id (int): Element ID.

        Returns:
            str: Formatted element set line.
        """
        return f"*Elset, elset=Set-{element_id}\n{element_id}\n"

    def write_mesh_to_file(self, X, Y):
        """
        Writes the mesh information to the Abaqus input file.

        Args:
            X (array): X coordinates of the mesh nodes.
            Y (array): Y coordinates of the mesh nodes.
        """
        # Compute node IDs and scaled coordinates
        node_ids, x_scaled, y_scaled = self.compute_node_data(X, Y, self.scale_factor)

        # Convert JAX arrays to NumPy arrays for file writing
        X_np, Y_np = np.array(X), np.array(Y)
        node_data = np.stack([node_ids.ravel(), x_scaled.ravel(), y_scaled.ravel()], axis=-1)

        def filter_nodes(condition):
            """Filters nodes based on a condition. 
            In this case, it is the boundary nodes where X or Y node coordinates are zero or maximum.
            """
            indices = np.where(condition)[0]
            return node_data[indices]

        # Identify boundary nodes based on their coordinates
        bottom_nodes = filter_nodes(Y_np.ravel() == 0)
        top_nodes = filter_nodes(Y_np.ravel() == self.y_max)
        left_nodes = filter_nodes(X_np.ravel() == 0)
        right_nodes = filter_nodes(X_np.ravel() == self.x_max)

        # Generate element connectivity based on node positions
        elements = []
        element_id = 1
        for i in range(X_np.shape[0] - 1):
            for j in range(X_np.shape[1] - 1):
                node1 = i * X_np.shape[1] + j + 1
                node2 = node1 + 1
                node3 = (i + 1) * X_np.shape[1] + j + 2
                node4 = node3 - 1
                elements.append((element_id, node1, node2, node3, node4))
                element_id += 1

        # Format the node and element data into strings
        node_lines = [f"{int(node_id)}, {x:.4f}, {y:.4f}\n" for node_id, x, y in node_data]
        element_lines = [f"{element_id}, {node1}, {node2}, {node3}, {node4}\n"
                         for element_id, node1, node2, node3, node4 in elements]

        # Generate node sets for boundary conditions
        node_sets = []
        for node_id, _, _ in bottom_nodes:
            node_sets.append(self.write_node(node_id, 'NODEBOTTOM'))
        for node_id, _, _ in left_nodes:
            node_sets.append(self.write_node(node_id, 'NODELEFT'))
        for node_id, _, _ in right_nodes:
            node_sets.append(self.write_node(node_id, 'NODERIGHT'))
        for node_id, _, _ in top_nodes:
            node_sets.append(self.write_node(node_id, 'NODETOP'))

        # Generate element sets for assigning material properties
        element_sets = [self.write_element(element_id) for element_id, _, _, _, _ in elements]

        # Write all data to the input file
        with open(self.file_path, 'w') as f:
            f.write("*Heading\n")
            f.write("** PARTS\n")
            f.write("*Part, name=PART-1\n")
            f.write("**\n")
            f.write("*NODE\n")
            f.writelines(node_lines)
            f.write("*Element, type=CPS4 \n")
            f.writelines(element_lines)

            # Write element sets with material properties
            for elset_name, element_id in zip(element_sets, range(1, len(elements) + 1)):
                f.write(elset_name)
                f.write(f"** Section: Set-{element_id}\n")
                f.write(f"*Solid Section, elset=Set-{element_id}, material=Material-{element_id}\n")

            f.write("*End Part\n")
            f.write("**\n")
            f.write("**\n")
            f.write("** ASSEMBLY\n")
            f.write("**\n")
            f.write("*Assembly, name=Assembly\n")
            f.write("**\n")
            f.write("*Instance, name=PART-1-1, part=PART-1\n")
            f.write("*End Instance\n")
            f.write("**\n")
            f.write("** NSETS\n")
            f.write("**\n")

            # Write node sets
            f.writelines(node_sets)
            f.write("*End Assembly\n")
            f.write("**\n")

        print("Mesh created successfully at:", self.file_path)


class MaterialInserter:
    """
    A class to insert material properties and boundary conditions into the Abaqus input file.
    
    The stress field is calculated based on Ludwik's equation:
        σ = σ_0 + K * (ε_p)^n

    Attributes:
        file_path (str): Path to the Abaqus input file.
        displacement_y (array): Array of y-direction displacements for nodes.
        displacement_x (array): Array of x-direction displacements for nodes.
        X (array): X coordinates of the mesh nodes.
        Y (array): Y coordinates of the mesh nodes.
        stress_field (array): Stress field data for each element.
    """

    def __init__(self, file_path, displacement_y, displacement_x, X, Y, stress_field):
        self.file_path = file_path
        self.displacement_y = displacement_y
        self.displacement_x = displacement_x
        self.X = X
        self.Y = Y
        self.stress_field = stress_field

    def insert_material_data(self):
        """
        Inserts material properties and boundary conditions into the Abaqus input file.
        """
        E_p = np.linspace(0.0, 0.005, 20)  # Plastic strain values
        num_x, num_y = self.stress_field.shape[1], self.stress_field.shape[2]
        material_data_lines = []

        # Write material properties for each element
        for i in range(num_x):
            for j in range(num_y):
                element_id = i * num_y + j + 1
                stress_values = self.stress_field[:, i, j]

                material_data_lines.append("** MATERIALS\n")
                material_data_lines.append(f"*Material, name=Material-{element_id}\n")
                material_data_lines.append("*Elastic\n")
                material_data_lines.append("205000., 0.25\n")  # Young's modulus and Poisson's ratio
                material_data_lines.append("*Plastic\n")

                stress_str = "\n".join([f"{stress_values[k]:.6f}, {E_p[k]:.6f}"
                                         for k in range(E_p.shape[0])])
                material_data_lines.append(f"{stress_str}\n")

        # Prepare boundary condition data
        boundary_data_lines = self._prepare_boundary_conditions(num_x, num_y)

        # Write data to the input file
        with open(self.file_path, 'a') as f:
            f.writelines(material_data_lines)
            f.write("** \n")
            f.write("** ----------------------------------------------------------------\n")
            f.write("** \n")
            f.write("** STEP: Step-1\n")
            f.write("** \n")
            f.write("*Step, name=Step-1, nlgeom=NO, inc=1000\n")
            f.write("*Static\n")
            f.write("0.001, 1., 1e-06, 1.\n")
            f.write("** \n")
            f.write("** BOUNDARY CONDITIONS\n")
            f.write("**\n")
            f.writelines(boundary_data_lines)
            f.write("**\n")
            f.write("** OUTPUT REQUESTS\n")
            f.write("**\n")
            f.write("*Restart, write, frequency=0\n")
            f.write("**\n")
            f.write("** FIELD OUTPUT: F-Output-1\n")
            f.write("**\n")
            f.write("*Output, field\n")
            f.write("*Node Output\n")
            f.write("RF, U\n")
            f.write("*Element Output, directions=YES\n")
            f.write("E, PE, PEEQ, PEMAG, S\n")
            f.write("**\n")
            f.write("** HISTORY OUTPUT: H-Output-1\n")
            f.write("**\n")
            f.write("*Output, history, variable=PRESELECT\n")
            f.write("*End Step\n")

        print("Material data inserted successfully into:", self.file_path)

    def _prepare_boundary_conditions(self, num_x, num_y):
        """
        Prepares boundary condition data for writing to the file.

        Args:
            num_x (int): Number of elements in the x-direction.
            num_y (int): Number of elements in the y-direction.

        Returns:
            list: Boundary condition lines for the input file.
        """
        # Extract boundary displacements from the displacement arrays
        bottom_displacements_y = self.displacement_y[:, 0]
        bottom_displacements_x = self.displacement_x[:, 0]

        top_displacements_y = self.displacement_y[:, -1]
        top_displacements_x = self.displacement_x[:, -1]

        left_displacement_y = self.displacement_y[0, :]
        left_displacement_x = self.displacement_x[0, :]

        right_displacements_y = self.displacement_y[-1, :]
        right_displacements_x = self.displacement_x[-1, :]
        
        # Generate interpolation points for finer resolution
        x_positions = np.arange(len(bottom_displacements_y))
        y_positions = np.arange(len(left_displacement_y))

        x_full = np.linspace(0, len(bottom_displacements_y) - 1, len(bottom_displacements_y) + 1)
        y_full = np.linspace(0, len(left_displacement_y) - 1, len(left_displacement_y) + 1)
       
        # Interpolate displacements for boundary nodes
        bottom_extra_y = np.interp(x_full, x_positions, bottom_displacements_y)
        bottom_extra_x = np.interp(x_full, x_positions, bottom_displacements_x)

        top_extra_y = np.interp(x_full, x_positions, top_displacements_y)
        top_extra_x = np.interp(x_full, x_positions, top_displacements_x)

        left_extra_y = np.interp(y_full, y_positions, left_displacement_y)
        left_extra_x = np.interp(y_full, y_positions, left_displacement_x)

        right_extra_y = np.interp(y_full, y_positions, right_displacements_y)
        right_extra_x = np.interp(y_full, y_positions, right_displacements_x)

        boundary_data_lines = []
        for j in range(len(bottom_extra_y)):
            bottom_node_id = j + 1
            boundary_data_lines.append(f"** Name: Bottom{bottom_node_id} Type: Displacement/Rotation\n*Boundary\n")
            boundary_data_lines.append(f"NodeBottom{bottom_node_id}, 1, 1, {bottom_extra_x[j]}\n")
            boundary_data_lines.append(f"NodeBottom{bottom_node_id}, 2, 2, {bottom_extra_y[j]}\n")

        for j in range(1, len(left_extra_y) - 1):
            left_node_id = (j * (num_x + 1)) + 1
            boundary_data_lines.append(f"** Name: Left{left_node_id} Type: Displacement/Rotation\n*Boundary\n")
            boundary_data_lines.append(f"NodeLeft{left_node_id}, 1, 1, {left_extra_x[j]}\n")
            boundary_data_lines.append(f"NodeLeft{left_node_id}, 2, 2, {left_extra_y[j]}\n")

        for j in range(1, len(left_extra_y) - 1):
            right_node_id = (j * (num_x + 1)) + 1 + num_x
            boundary_data_lines.append(f"** Name: Right{right_node_id} Type: Displacement/Rotation\n*Boundary\n")
            boundary_data_lines.append(f"NodeRight{right_node_id}, 1, 1, {right_extra_x[j]}\n")
            boundary_data_lines.append(f"NodeRight{right_node_id}, 2, 2, {right_extra_y[j]}\n")

        for j in range(len(top_extra_y)):
            top_node_id = (num_y * (num_x + 1)) + (j + 1)
            boundary_data_lines.append(f"** Name: Top{top_node_id} Type: Displacement/Rotation\n*Boundary\n")
            boundary_data_lines.append(f"NodeTop{top_node_id}, 1, 1, {top_extra_x[j]}\n")
            boundary_data_lines.append(f"NodeTop{top_node_id}, 2, 2, {top_extra_y[j]}\n")

        return boundary_data_lines
