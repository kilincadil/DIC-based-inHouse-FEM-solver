"""
@author: akilinc

Abaqus Input File Reader

This script parses an Abaqus mesh input file to extract node coordinates, element definitions,
and element sets (elsets). It supports reading:

1. Nodes: Extracting node IDs with their X and Y coordinates.
2. Elements: Extracting element IDs and their associated node connectivity.
3. Elsets: Extracting named sets of elements, including those defined by explicit lists and generated ranges. In our case, elsets represent the grains-twins.
"""

class AbaqusInputReader:
    def __init__(self, file_path):
        """
        Initializes the AbaqusInputReader with the input file path.

        Parameters:
        file_path (str): Path to the Abaqus .inp file.
        """
        self.file_path = file_path
        self.nodes = []
        self.elements = []
        self.elsets = {}

    def read_nodes(self):
        """
        Parses the node section from the Abaqus input file.

        Extracts node IDs and their X, Y coordinates.
        """
        with open(self.file_path, 'r') as file:
            read_nodes = False
            for line in file:
                line = line.strip()
                if line.lower().startswith('*node'):
                    read_nodes = True
                    continue
                elif line.startswith('*'):
                    read_nodes = False
                if read_nodes and line:
                    parts = line.split(',')
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    self.nodes.append((node_id, x, y))

    def read_elements(self):
        """
        Parses the element section from the Abaqus input file.

        Extracts element IDs and their connected nodes.
        """
        with open(self.file_path, 'r') as file:
            read_elements = False
            for line in file:
                line = line.strip()
                if line.lower().startswith('*element'):
                    read_elements = True
                    continue
                elif line.startswith('*'):
                    read_elements = False
                if read_elements and line:
                    parts = line.split(',')
                    element_id = int(parts[0])
                    element_nodes = [int(n.strip()) for n in parts[1:]]
                    self.elements.append((element_id, *element_nodes))

    def read_elsets(self):
        """
        Parses the elset section from the Abaqus input file.

        Supports both explicit element lists and generated element ranges.
        """
        with open(self.file_path, 'r') as file:
            read_generate = False
            elset_name = None
            for line in file:
                line = line.strip()
                if line.lower().startswith('*elset, elset=face_') or line.lower().startswith('*elset,elset=face_'):
                    elset_info = line.split(',')[1].strip()
                    elset_name = elset_info.split('=')[1].split()[0]
                    self.elsets[elset_name] = []

                    if 'generate' in line:
                        read_generate = True
                    else:
                        read_generate = False
                    continue

                if read_generate:
                    generate_info = line.strip().split(',')
                    start = int(generate_info[0])
                    end = int(generate_info[1])
                    step = int(generate_info[2])
                    self.elsets[elset_name].extend(range(start, end + 1, step))
                    read_generate = False
                elif line and not line.startswith("*"):
                    if elset_name is not None:
                        elset_elements = [int(e.strip()) for e in line.split(',') if e.strip().isdigit()]
                        self.elsets[elset_name].extend(elset_elements)

    def get_nodes(self):
        """
        Returns the parsed node data.

        Returns:
        list: A list of tuples, where each tuple contains (node_id, x, y).
        """
        return self.nodes

    def get_elements(self):
        """
        Returns the parsed element data.

        Returns:
        list: A list of tuples, where each tuple contains (element_id, node1, node2, ...).
        """
        return self.elements

    def get_elsets(self):
        """
        Returns the parsed elset data.

        Returns:
        dict: A dictionary where keys are elset names and values are lists of element IDs.
        """
        return self.elsets

""" #Example use
input_file_path = r'path/to/input.inp'
reader = AbaqusInputReader(input_file_path)
reader.read_nodes()
reader.read_elements()
reader.read_elsets()

#For multiple use
nodes = reader.get_nodes()
elements = reader.get_elements()
elsets = reader.get_elsets()
"""