"""
Iterative indirect immunofluorescent imaging toolbox | David Cullen | Jones Lab 2023
"""
import numpy as np
import pandas as pd

from pystackreg import StackReg
def register(image: np.ndarray) -> np.ndarray:
    """
    Register an image stack using StackReg.

    Parameters:
    image (np.ndarray): The image stack to be registered.

    Returns:
    np.ndarray: The registered image stack.
    """
    sr = StackReg(StackReg.RIGID_BODY)
    return sr.register_stack(image, reference='first')

def transform(image: np.ndarray, tmats: np.ndarray) -> np.ndarray:
    """
    Transforms the given image using the specified transformation matrices.

    Parameters:
    img (np.ndarray): The image to be transformed.
    tmats (np.ndarray): The transformation matrices to apply to the image.

    Returns:
    np.ndarray: The transformed image.
    """
    sr = StackReg(StackReg.RIGID_BODY)
    return sr.transform(image, tmats)

def get_transformation_4i(path: str) -> dict[tuple[str, str, str, int], np.ndarray]:
    """
    Get the transformations for all fields in the microscopy plate.
    Parameters:
    path: Path to the image directory.
    
    Returns:
    Dictionary of transformations for each field, with the keys being tuples (iteration, row, column, field).
    """
    tmats = {}
    DAPI_metadata = get_metadata(path, groups = ['Row', 'Column', 'Field'], only_DAPI=True)
    for name, group in DAPI_metadata:
        image = assemble_image(group, path)
        tmat = register(image)
        tmats[(str(name[0]), str(name[1]), str(name[2]))] = tmat 
        
    transformation_metadata = {}
    metadata = get_metadata(path, groups = ['Row', 'Column', 'Field'])
    for name, group in metadata:
        current_field_tmats = tmats[(str(name[0]), str(name[1]), str(name[2]))]
        for i in group['Iteration']:
            transform = current_field_tmats[int(i)]
            transformation_metadata[(int(i), str(name[0]), str(name[1]), str(name[2]))] = transform
    return transformation_metadata

def import_one_image(group: pd.DataFrame, path: str) -> np.ndarray:
    """
    Import a single image from file.

    Parameters:
        group (pd.DataFrame): The metadata for the image to be imported.
        path (str): The path to the directory containing the images.

    Returns:
        np.ndarray: The imported 2D image as a NumPy array.
    """
    array = tf.imread(str(path) + "/" + group.String)
    return array

def get_nuc(path: str, transformation_metadata: dict[tuple[int, str, str, str], np.ndarray], group: pd.DataFrame, name: tuple[str, str, str]) -> np.ndarray:
    '''
    Get nuclei from images for the current field of view, after transforming the image according to transformation metadata.
    
    Parameters:
    path (str): The path to the folder containing the images.
    transformation_metadata (Dict[Tuple[int, str, str, str], np.ndarray]): The dictionary of transformation matrices for each field of view.
    group (pd.DataFrame): The DataFrame containing metadata for a specific field of view.
    name (Tuple[str, str, str]): The name of the field of view in the form of (row, column, field).
    
    Returns:
    np.ndarray: The segmented nuclei image.
    '''
    first_iteration = group[group.Iteration == group.Iteration.min()] # Select only the first iteration
    first_DAPI = first_iteration[first_iteration['WV'] == '405'] # Select only the DAPI images
    current_metadata = tuple((first_DAPI.iloc[0]['Iteration'], str(name[0]), str(name[1]), str(name[2]))) # Get the metadata for the first DAPI image
    DAPI = import_one_image(first_DAPI, path) 
    # Import the first DAPI image
    transformation = transformation_metadata[current_metadata] # Get the transformation matrix for this field of view
    registered = transform(DAPI, tmats = transformation) # Register the image
    return tools.segment_nuclei(registered) # Segment the nuclei

    