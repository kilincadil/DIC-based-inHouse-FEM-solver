import os
import numpy as np

class ArrayLoader:
    """
    The ArrayLoader class loads multiple '.npy' files from a specified folder, 
    organizes them in ascending order based on their filenames (assumed to be numeric), 
    and returns them as a list of NumPy arrays.
    """
    
    def __init__(self, folder_path):
        """
        Initializes the `ArrayLoader` object with the path to the folder containing '.npy' files.

        Parameters:
        - folder_path (str): The directory path where the '.npy' files are stored.
        """
        self.folder_path = folder_path

    def load(self):
        """
        Loads all '.npy' files from the specified folder, sorts them based on their filenames (numerically), 
        and returns a list of NumPy arrays corresponding to the loaded data.
        
        The filenames are expected to be numeric (e.g., '1.npy', '2.npy', etc.), and the files will be loaded 
        in ascending order of these numeric values.
        
        Returns:
        - list: A list of NumPy arrays, one for each '.npy' file found in the specified folder.
        """
        npy_files = [f for f in os.listdir(self.folder_path) if f.endswith('.npy')]
        npy_files.sort(key=lambda x: int(x.split('.')[0]))

        arrays = []

        for npy_file in npy_files:
            file_path = os.path.join(self.folder_path, npy_file)
            array_data = np.load(file_path)
            arrays.append(array_data)

        return arrays
