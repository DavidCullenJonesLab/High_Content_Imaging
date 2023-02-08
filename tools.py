"""
Image processing toolbox | David Cullen | Jones Lab 2023
"""
import numpy as np
import pandas as pd

import os
import shutil
def folders() -> None:
    import os
    ''' Create required folders, removing existing ones if necessary. '''
    for folder in ['Quality Control', 'Quality Control/Rejects', 'Images']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

def extract_file_info(file_name: str, is_4i: bool = False) -> dict[str, str]:
    """
    Extracts the file information from the file name using regular expression.

    Parameters:
        file_name (str): The file name to extract information from.
        is_4i (bool, optional): A flag to indicate if the file is from an iterative 
        staining experiment (default is False).

    Returns:
        Dict[str, str]: A dictionary with keys as the extracted file information and values 
        as the corresponding information. The regex used is aimed to extract information from 
        Incell Analysier 6500 tifs.
    """
    if not is_4i:
        pattern = '(?P<Row>[a-zA-Z]) - (?P<Column>[0-9]+)\(fld (?P<Field>\d) wv (?P<WV>[0-9]+) - (?P<Filter>[a-zA-Z]+)\)'
    else:
        pattern = '(?P<Iteration>[0-9]+)_(?P<Row>[a-zA-Z]) - (?P<Column>[0-9]+)\(fld (?P<Field>\d) wv (?P<WV>[0-9]+) - (?P<Filter>[a-zA-Z]+)\)'
    match = re.match(pattern, file_name)
    if match is None:
        return {}
    return match.groupdict()

import re
def get_metadata(path: str, groups: list[str], only_DAPI: bool = False) -> pd.DataFrame:
    """
    Retrieve metadata information for files in a directory and return the grouped metadata.

    Parameters:
    path (str): Path to the directory where the files are located.
    groups (List[str]): List of column names to use for grouping the metadata.
    only_DAPI (bool, optional): Only return metadata for images with a WV of 405 nm (default is False). For 4i registration.

    Returns:
    pd.DataFrame: A dataframe containing the grouped metadata information.
    """
    file_list = [entry.name for entry in os.scandir(path) if entry.is_file()]
    metadata = [{'String': file_name, **extract_file_info(file_name)} for file_name in file_list]
    metadata = pd.DataFrame(metadata)
    if only_DAPI:
        metadata = metadata[metadata['WV'] == '405']
    return metadata.groupby(groups)

import tifffile as tf
def assemble_image(name: tuple[str, str, str], group, path: str) -> np.ndarray:
    '''
    Import each file in a group and assemble a multi-channel image.

    Parameters:
        name (Tuple[str, str, str]): A tuple of 3 strings.
        group (GroupType): An object of GroupType, containing a list of strings.
        path (str): A file path to the directory where the image files are located.

    Returns:
        np.ndarray: A multi-channel image as a numpy array.
    '''
#    print(f"\rWell {name[0]+name[1]}, Field {name[2]}")
    arrays = [tf.imread(str(path) + "/" + each_string) for each_string in group.String]
    return np.array(arrays)



from cellpose import models
from skimage.segmentation import clear_border
def segment_nuclei(image: np.ndarray) -> np.ndarray:
    '''
    Use pre-trained model to segment nuclei in an image. 
    This is the default function, for 10x images widefield.

    Parameters:
        image (np.ndarray): A single channel image.

    Returns:
        np.ndarray: A binary mask of the segmented nuclei.
    '''
    model = models.CellposeModel(model_type='cellpose7')
    masks = model.eval(image[0], channels=[0,0])
    masks = clear_border(masks[0])
    #print("Nuclei found")
    return masks

def segment_nuclei_60x(image: np.ndarray) -> np.ndarray:
    '''
    Use pre-trained model to segment nuclei in an image. 
    This is the altered function, for 60x images widefield.


    Parameters:
        image (np.ndarray): A single channel image.

    Returns:
        np.ndarray: A binary mask of the segmented nuclei.
    '''
    model = models.CellposeModel(model_type='cellpose7')
    masks = model.eval(image[0], diameter=160, channels=[0,0])
    masks = clear_border(masks[0])
    print("Nuclei found")
    return masks

def remove_background(image: np.ndarray, threshold: int = 500) -> np.ndarray:
    '''
    This function removes the background from a 3D image by subtracting the mean 
    of the pixels with values below a certain threshold from each channel of the image.

    Parameters:
    - image (numpy.ndarray): a 3D array representing the image to process. 
    Shape: (N, H, W), where N is the number of channels, H is the height, and W is the width.
    - threshold (int, optional): a value that determines the cutoff for the pixels to be 
    considered as background. Default: 500.
    
    Returns:
    - result (numpy.ndarray): a 3D array with the same shape as `image`, 
    but with the background subtracted.
    '''
    result = np.zeros_like(image)
    for i, channel in enumerate(image):
        background_value = np.mean(channel[channel < threshold])
        result[i] = np.maximum(0, channel - background_value)
    return result

def output_image(name: str, meta: tuple[str, str, str], image: np.ndarray, masks: np.ndarray) -> None:
    '''
    Save example images with nuclear rings.

    Parameters:
        name: str, name of the output file
        meta: Tuple[str, str, str], metadata of the image
        image: np.ndarray, input multi-channel image
        masks: np.ndarray, input 2D masks image

    Returns:
        None, saves the modified image to a file in the Images folder
    '''
    # Compute the inner ring of the masks
    rings = inner_ring(masks, erosion = 1)
    # Copy the input image
    output_image = image.copy()
    # Loop through each channel in the output image
    for channel in output_image:
        # Modify the current channel using np.putmask
        np.putmask(channel, rings, rings+10000)
    # Subtract 65536 from the image
    output_image = 65536 - output_image
    # Write the output image to a file
    tf.imwrite('Images/' + str(name) + str(meta[0]) + str(meta[1]) + str(meta[2]) + '.ome.tif', output_image)

def erode(masks: np.ndarray, erosion: int) -> np.ndarray:
    '''
    Erode the mask.

    Parameters:
        masks (np.ndarray): The masks to erode.
        erosion (int): The size of the erosion to perform.

    Returns:
        np.ndarray: The eroded masks.
    '''
    # Create a square kernel with size (erosion*2+1, erosion*2+1) and data type np.uint8
    kernel = np.ones((erosion*2+1, erosion*2+1), np.uint8)
    # Convert masks to np.ndarray with data type np.uint16
    masks = np.asarray(masks, dtype="uint16")
    # Erode the masks with the kernel using OpenCV's cv2.erode function
    return cv2.erode(masks, kernel)

def inner_ring(masks: np.ndarray, erosion: int) -> np.ndarray:
    '''
    Create a ring on the inside border of the mask.

    Parameters:
        masks (np.ndarray): The masks to create a ring on.
        erosion (int): The size of the erosion to perform.

    Returns:
        np.ndarray: The inside ring of the masks.
    '''
    # Erode the masks
    eroded = erode(masks, erosion)
    # Return the result of original masks minus the eroded masks.
    return masks - eroded

def outer_ring(masks: np.ndarray, dilation: int, gap: int) -> np.ndarray:
    ''' 
    Create a ring outside the border of the mask. 
    
    Parameters:
        masks: np.ndarray, the binary image masks.
        dilation: int, the size of the dilation.
        gap: int, the size of the gap.

    Returns:
        np.ndarray, the image with the outer ring.
        
    ''' 
    # Define the dilation kernel
    kernel = np.ones((dilation*2+1, dilation*2+1), np.uint8)
    # Define the gap kernel
    gap_kernel = np.ones((gap*2+1, gap*2+1), np.uint8)
    # Convert the masks to a uint8 array
    masks = np.asarray(masks, dtype="uint8")
    # Perform a slight dilation on the masks using the dilation kernel
    slight_dilation = cv2.dilate(masks, kernel)
    # Perform a dilation on the masks using the gap kernel
    dilated = cv2.dilate(masks, gap_kernel)
    # Return the difference between the dilated and slight_dilation masks
    return dilated - slight_dilation

def output_image(name: str, meta: dict, image: np.ndarray, nuclear_masks: np.ndarray):
    """
    Add rings around nuclei and output image to disk and save metadata.

    Args:
        name (str): name of the image to save.
        meta (dict): metadata to save.
        image (np.ndarray): image data.
        nuclear_masks (np.ndarray): nuclear masks data.

    Returns:
        None
    """
    
    rings = inner_ring(nuclear_masks, erosion = 1)
    output_image = image.copy()
    for channel in output_image:
        np.putmask(channel, rings, rings+10000)
    output_image = 65536 - output_image
    tf.imwrite('Images/' + str(name) + str(meta[0]) + str(meta[1]) + str(meta[2]) + '.ome.tif', output_image)
    
def output_rgb(image: np.ndarray, name: list[str]) -> None:
    '''
    Save the image in RGB format.

    Parameters:
    image: a numpy array with shape (3, 2040, 2040)
        The input image.
    name: a list of 3 strings
        The name of the image.

    Returns:
    None. The outpput image will be saved in rgb folder.
    '''
    # Initialize the RGB image with zeros
    rgb = np.zeros(shape=(2040, 2040, 3), dtype=np.uint16)
    # Set the red channel of the RGB image
    rgb[:, :, 0] = image[0]
    # Set the green channel of the RGB image
    rgb[:, :, 1] = image[1]
    # Save the RGB image to disk
    tf.imsave('rgb/' + 'Well_' + name[0]+name[1] + '_Field_' + name[2] + '.tif', rgb)

def crop_image(image: np.ndarray, centroid: tuple[int, int], crop_size: int = 40) -> np.ndarray:
    '''
    Crop an image around a given centroid to a specified size. This is used to generate equally-sized cropped images of nuclei.

    Parameters:
    image (numpy.ndarray): The input image to be cropped. 
                           The shape should be (rows, cols, channels).
    centroid (tuple): The centroid (row, col) around which to crop the image.
    crop_size (int, optional): The size (height and width) of the cropped image. 
                               Defaults to 40.

    Returns:
    numpy.ndarray: The cropped and padded image with shape (crop_size, crop_size, channels).
    '''
    # Get the shape of the image
    rows, cols, channels = image.shape

    # Calculate the start and end row and column indices for the crop
    start_row = max(0, int(centroid[0] - crop_size / 2))
    end_row = min(rows, int(centroid[0] + crop_size / 2))
    start_col = max(0, int(centroid[1] - crop_size / 2))
    end_col = min(cols, int(centroid[1] + crop_size / 2))
    
    # Crop the image
    cropped_image = image[start_row:end_row, start_col:end_col, :]
    
    # Calculate the padding for each side if the cropped image is not the correct size
    top_padding = bottom_padding = int((crop_size - cropped_image.shape[0]) / 2)
    left_padding = right_padding = int((crop_size - cropped_image.shape[1]) / 2)
    if cropped_image.shape[0] < crop_size:
        bottom_padding = bottom_padding + (crop_size - cropped_image.shape[0]) % 2
    if cropped_image.shape[1] < crop_size:
        right_padding = right_padding + (crop_size - cropped_image.shape[1]) % 2
    
    # Pad the cropped image to the required size
    padded_image = np.pad(cropped_image, ((top_padding, bottom_padding), (left_padding, right_padding), (0, 0)), 'constant')
    
    return padded_image

def intensity_sum(regionmask: np.ndarray, intensity_image: np.ndarray) -> float:
    '''
    Return the total intensity for all pixels inside the mask.

    Parameters
    ----------
    regionmask : np.ndarray
        A binary mask indicating which pixels belong to the region of interest.
    intensity_image : np.ndarray
        The 2D intensity image corresponding to the regionmask.

    Returns:
    float
        The sum of intensities for all pixels inside the regionmask.
    '''
    return np.sum(intensity_image[regionmask])

def intensity_mean_eroded(regionmask: np.ndarray, intensity_image: np.ndarray) -> float:
    '''
    Erode the regionmask, then find the mean intensity of each channel within.
    
    Parameters:
    - regionmask (np.ndarray): The binary mask representing the region of interest.
    - intensity_image (np.ndarray): The multi-channel intensity image.
    
    Returns:
    - float: The mean intensity of the eroded regionmask.
    '''
    eroded = erode(regionmask, erosion = 1)
    eroded_image = intensity_image[eroded]
    return np.mean(eroded_image)

def measure_properties(name: tuple[str, str, int], image: np.ndarray, regions: list, results: list) -> list[tuple]:
    '''
    Measure properties of nuclei in an image.
    
    Parameters:
    - name: Tuple of three elements representing the well, row, and column of the current image
    - image: numpy ndarray representing the image to be processed
    - regions: list of regionprops objects representing the nuclei in the image
    
    Returns:
    - results: a list of lists where each sublist represents the properties of a single nucleus
    
    The properties included in each sublist are:
    - well: well identifier, a concatenation of the row and column in the form "RowColumn"
    - field: identifier of the field within the well
    - label: identifier of the nucleus within the field
    - area: area of the nucleus, in pixels
    - circularity: measure of circularity, defined as 4 * pi * area / perimeter^2
    - Mean_DAPI: the mean log10 intensity of the nucleus in the DAPI channel
    - Mean_Green (optional): the mean log10 intensity of the nucleus in the Green channel, if present
    - Mean_Orange (optional): the mean log10 intensity of the nucleus in the Orange channel, if present
    - Mean_EdU (optional): the mean log10 intensity of the nucleus in the EdU channel, if present
    - Sum_DAPI: the sum log10 intensity of the nucleus in the DAPI channel
    - Sum_Green (optional): the sum log10 intensity of the nucleus in the Green channel, if present
    - Sum_Orange (optional): the sum log10 intensity of the nucleus in the Orange channel, if present
    - Sum_EdU (optional): the sum log10 intensity of the nucleus in the EdU channel, if present
    '''
    #results = []
    for props in regions:
        circularity = 4 * np.pi * props.area / (props.perimeter**2)
        result = [name[0]+name[1], name[2], props.label, props.area, circularity] + [np.log10(i) for i in props.intensity_mean] + [np.log10(i) for i in props.intensity_sum]
        results.append(result)
#    return results
import pandas as pd    

    
def write_df(results: list, image: np.ndarray, dhb: list) -> None:
    '''
    Write dataframe, merge it with the plate map and export as a csv.
    
    Parameters:
    - results (List[List[Union[str, int, float]]]): A 2D list containing the well, field, label, area, circularity, 
      mean and sum values of each color in the image.
    - image (np.ndarray): A numpy array representing the image.
    - dhb (List[float]): A list of floating point values for the dhb measurement for each nucleus.
    
    Returns:
    - None
    '''
    base_list = ['Well', 'Field', 'Label', 'Area', 'Circularity']
    if image.shape[0] == 4:
        df = pd.DataFrame(results, columns = base_list + ['Mean_DAPI', 'Mean_Green', 'Mean_Orange', 'Mean_EdU','Sum_DAPI', 'Sum_Green', 'Sum_Orange', 'Sum_EdU'])
    elif image.shape[0] == 3:
        df = pd.DataFrame(results, columns = base_list + ['Mean_DAPI', 'Mean_Green', 'Mean_EdU','Sum_DAPI', 'Sum_Green', 'Sum_EdU'])
    else:
        df = pd.DataFrame(results, columns = base_list + ['Mean_DAPI', 'Mean_EdU','Sum_DAPI', 'Sum_EdU'])
    df['DHB'] = dhb
    df.to_csv('df.csv')
        
def process_images(path):
    '''
    1. tools.folders(): 
    This creates a set of directories that will be used to store the processed images.

    2. metadata = tools.get_metadata(path, groups = ['Row', 'Column', 'Field']): This retrieves 
    metadata about each image file in the path directory and groups the images based on their row, 
    column, and field information.

    3. for name, group in metadata:: This loop iterates over each group of images and performs the 
    following steps for each group.

    4. image = tools.assemble_image(name, group, path): This function takes the grouped images and 
    combines them into a single 3D image.
    
    4b. image = tools.remove_background(unprocessed_image): This function takes the 3D image created
    in step 4 and subtracts from every pixel the mean intensity of the background.
    
    5. nuclear_masks = tools.segment_nuclei(image): This function segments the nuclei in the image,
    creating a binary mask of the nuclei.

    6. cyto_masks = ktr.segment_cytoplasm(ktr.rgb(image)): This function segments the cytoplasm in
    the image, creating a binary mask of the cytoplasm.

    7. regionprops_image = np.moveaxis(image, 0, -1): This function moves the intensity information
    from the
    first axis to the last axis of the image, so that it is compatible with the regionprops 
    function.

    8. nuclear_props = regionprops(nuclear_masks, intensity_image=regionprops_image,
    extra_properties=(tools.intensity_sum,tools.intensity_mean_eroded,)): This function uses the
    regionprops 
    function from the skimage library to measure various properties of the nuclei, including area,
    mean intensity, and sum of intensities.

    9. cyto_props = regionprops(cyto_masks, intensity_image=regionprops_image): This function
    measures various properties of the cytoplasm, including area and mean intensity.

    10. dhb = ktr.cytoplasmic_nuclear_ratio(nuclear_props, cyto_props, nuclear_masks, cyto_masks, 
    regionprops_image, name): This function calculates the cytoplasmic-to-nuclear intensity ratio
    for each nucleus.

    11. results = tools.measure_properties(name, image, nuclear_props): This function measures
    various properties of the nuclei, including area, circularity, and mean and sum intensities.

    12. tools.write_df(results, image, dhb): This function writes the results to a csv file and
    includes the cytoplasmic-to-nuclear intensity ratio as a new column.
    '''
    results = []
    dhb = []
    tools.folders()
    metadata = tools.get_metadata(path, groups = ['Row', 'Column', 'Field'])
    for name, group in metadata:
        image = tools.assemble_image(name, group, path)
        nuclear_masks = tools.segment_nuclei(image)
        cyto_masks = ktr.segment_cytoplasm(ktr.rgb(image))
        regionprops_image = np.moveaxis(image, 0, -1)
        nuclear_props = regionprops(nuclear_masks, intensity_image=regionprops_image, extra_properties=(tools.intensity_sum,tools.intensity_mean_eroded,))
        cyto_props = regionprops(cyto_masks, intensity_image=regionprops_image)
        ktr.cytoplasmic_nuclear_ratio(nuclear_props, cyto_props, nuclear_masks, cyto_masks, regionprops_image, name)
        tools.measure_properties(name, image, nuclear_props)
    tools.write_df(results, image, dhb)
        
