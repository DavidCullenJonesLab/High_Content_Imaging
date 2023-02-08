"""
Kinase translocation reporter toolbox | David Cullen | Jones Lab 2023
"""

def rgb(image):
    rgb = np.zeros(shape=(2040,2040,3),dtype=np.uint16)
    rgb[:,:,0] = image[0]
    rgb[:,:,1] = image[1]
    return rgb

from cellpose import models
def segment_cytoplasm(rgb):
    ''' 
    Segment cytoplasm in the input RGB image. 
    
    Parameters:
    rgb (ndarray): RGB image with shape (rows, cols, channels)
    
    Returns:
    cyto_mask (ndarray): Binary image of cytoplasm segmentation with shape (rows, cols)
    '''
    # Load the pre-trained model for cytoplasm segmentation
    model = models.CellposeModel(model_type='ringpose_230402')
    
    # Evaluate the model on the input RGB image
    cyto_masks = model.eval(rgb, channels=[2,1])
    # Return the binary mask of the cytoplasm segmentation
    #print("Cytoplasm found")
    return cyto_masks[0]

from scipy.spatial.distance import cdist 
def get_closest_cyto_indices(nuclear_props, cyto_props):
    if len(nuclear_props) == 0 or len(cyto_props) == 0:
        # Return an empty array if there are no cells present in the field of view
        return np.array([])
    '''
    This function takes in two lists of region properties of the nuclei and cytoplasm and returns the indices of the closest cytoplasmic regions to each nucleus.

    Parameters:
    - nuclear_props (list): a list of `regionprops` objects, representing the properties of each nucleus in the image
    - cyto_props (list): a list of `regionprops` objects, representing the properties of each cytoplasmic region in the image

    Returns:
    - closest_cyto_indices (numpy.ndarray): an array of shape len(nuclear_props), where each element represents the index of the closest cytoplasmic region to the corresponding nucleus.
    '''
    # Get the centroid coordinates of each nucleus and cytoplasmic region
    nuclear_coords = [prop.centroid for prop in nuclear_props]
    cyto_coords = [prop.centroid for prop in cyto_props]
    # Calculate the pairwise distances between the nuclei and cytoplasmic regions
    distances = cdist(nuclear_coords, cyto_coords)
    # Get the indices of the closest cytoplasmic region to each nucleus
    closest_cyto_indices = np.argmin(distances, axis=1)

    return closest_cyto_indices

import numpy as np
import cv2
import tifffile as tf
def cytoplasmic_nuclear_ratio(nuclear_props: list, cyto_props: list, nuclear_masks: np.ndarray, cyto_masks: np.ndarray, intensity_image: np.ndarray, name: tuple[str, str, str], dhb: list) -> float:
    
    """
    Given a set of nuclear properties and cytoplasmic properties, this function calculates the ratio of 
    the mean intensity of cytoplasmic regions around each nucleus to the mean intensity of the nuclei.

    Parameters
    ----------
    nuclear_props: list
        List of skimage.measure._regionprops._RegionProperties objects representing the properties of nuclei.
    cyto_props: list
        List of skimage.measure._regionprops._RegionProperties objects representing the properties of cytoplasmic regions.
    intensity_image: numpy.ndarray
        Numpy array representing the intensity image.
    cyto_masks: numpy.ndarray
        Binary numpy array representing the cytoplasmic masks.

    Returns
    -------
    dhb: float
        The ratio of cytoplasmic to nuclear mean intensity for each nucleus.
    """
    closest_cyto_indices = get_closest_cyto_indices(nuclear_props, cyto_props)
    dilated_nuclei = cv2.dilate(nuclear_masks, np.ones((5,5), np.uint8))
    ring_masks = np.where(dilated_nuclei==0, cyto_masks, 0)
    st = np.stack((ring_masks, nuclear_masks))#, intensity_image[:,:,1])
    sta = np.max(st, axis=0)
    tf.imwrite('Images/' + name[0] + '_' + name[1] + '_' + name[2] + '_' + 'stac.tif', sta)
    output_image = intensity_image[:,:,1].copy()
    output = output_image + ring_masks
    np.putmask(output_image, ring_masks, ring_masks+10000)
    
    tf.imwrite('Images/' + name[0] + '_' + name[1] + '_' + name[2] + '_' + 'ringon.tif', output_image)
    tf.imwrite('Images/' + name[0] + '_' + name[1] + '_' + name[2] + '_' + 'output.tif', output)
    #dhb = []
    for i, nuclear_prop in enumerate(nuclear_props):
        closest_cyto_index = closest_cyto_indices[i]
        cyto_prop = cyto_props[closest_cyto_index]
        ring_indices = np.where(ring_masks == cyto_prop.label)
        green = intensity_image[:,:,1]
        ring_pixels = green[ring_indices]
        ring_mean_intensity = np.mean(ring_pixels)
        nuclear_intensity = nuclear_prop.intensity_mean[1]
        ratio = ring_mean_intensity / nuclear_intensity
        #ring_pixels = cyto_masks[cyto_prop.label == nuclear_prop.label]
        #ring_mean_intensity = np.mean(intensity_image[ring_pixels])
        dhb.append(ratio)
        #print(f"Nucleus {nuclear_prop.label}, cyt {cyto_prop.label}, at {nuclear_prop.centroid}, {cyto_prop.centroid}")
    return dhb


    