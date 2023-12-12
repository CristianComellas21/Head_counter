# %%
from matplotlib import pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from lib.evaluate import evaluate_method
import pandas as pd

REFERENCE_IMAGE = cv2.imread("images/1660626000.jpg")
REFERENCE_IMAGE = cv2.cvtColor(REFERENCE_IMAGE, cv2.COLOR_BGR2RGB)
GRAY_REFERENCE_IMAGE = cv2.cvtColor(REFERENCE_IMAGE, cv2.COLOR_RGB2HSV)[:, :, 0]

# --- Auxiliary functions ---

def __plot_debug(image, debug=False):

    if not debug:
        return
    
    # Plot the image
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()



# --- Method functions ---

def __create_mask(image, debug=False):
    """
    Create a mask for the image.
    """
    # __plot_debug(image, debug)

    # Create a mask with the same size as the image and one channel, filled with ones
    mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Remove the top part of the image, where the sky and the boats are
    mask[0:450] = 0

    # Remove the wall at the left
    mask[450:580, 0:20] = 0
    mask[450:540, 20:35] = 0
    mask[450:530, 35:100] = 0


    return mask





# %% 
MASK = __create_mask(REFERENCE_IMAGE, debug=True)

# --- Main function ---
def locate_people(image, threshold=100, min_area=800, max_area=15000):
    """
    Locate people in an image.
    """
    # __plot_debug(image, debug)



    # Convert the image to grayscale
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # # Alternative gray conversion for Reference Image
    # ref_image = cv2.cvtColor(REFERENCE_IMAGE, cv2.COLOR_RGB2HSV)
    # ref_image = ref_image[:, :, 2]

    # ref_image *= MASK

    # __plot_debug(ref_image, debug)

    # Alternative gray conversion using HSV
    image_aux = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = image_aux[:, :, 0], image_aux[:, :, 1], image_aux[:, :, 2]



    # Take blue watter from the image using mask from H of HSV
    water_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    water_mask[(h > 70) & (h < 140) & (s > 50)] = 0
    water_mask = cv2.erode(water_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    
    # Take sand from the image using mask from H of HSV
    sand_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    sand_mask[(h > 0) & (h < 25) & (s < 30)] = 0
    sand_mask = cv2.erode(sand_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)


    mask = MASK.copy()
    mask *= water_mask
    mask *= sand_mask



    # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1]
    
    diff_image = np.abs(gray_image - GRAY_REFERENCE_IMAGE)


    thresholded_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    thresholded_image[diff_image > threshold] = 255

    # Apply mask to thresholded image
    thresholded_image *= mask

    canny_image = cv2.Canny(thresholded_image, 100, 200)


    # Apply close operation to thresholded image
    canny_image = cv2.morphologyEx(canny_image, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)


    # Count connected components
    (n_components, labels, stats, centroids) = cv2.connectedComponentsWithStats(canny_image)
    

    x = np.zeros((n_components), dtype=np.int32)
    y = np.zeros((n_components), dtype=np.int32)

    n_components = 0
    for label in range(1, labels.max() + 1):
        area = stats[label, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            cX, cY = centroids[label]
            x[n_components] = int(cX)
            y[n_components] = int(cY)
            n_components += 1

    return pd.DataFrame({
        'x': x[:n_components],
        'y': y[:n_components]
    })



# %%
