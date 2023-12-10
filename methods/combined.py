# %%
from matplotlib import pyplot as plt
import cv2
import numpy as np
from lib.nms import non_max_suppression, aspect_ratio_filter
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
def locate_people(image, threshold=100, stride=90, patch_size=(200, 200), aspect_ratio_threshold=1.5):
    """
    Locate people in an image.
    """
    image_aux = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = image_aux[:, :, 0], image_aux[:, :, 1], image_aux[:, :, 2]

    # Take blue watter from the image using mask from H of HSV
    water_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    water_mask[(h > 70) & (h < 140) & (s > 50)] = 0
    water_mask = cv2.erode(water_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    # Take sand from the image using mask from H of HSV
    sand_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    sand_mask[(h > 0) & (h < 40) & (s < 37)] = 0
    sand_mask = cv2.erode(sand_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)

    mask = MASK.copy()
    mask *= water_mask
    mask *= sand_mask

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 0]

    diff_image = np.abs(gray_image - GRAY_REFERENCE_IMAGE)

    thresholded_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    thresholded_image[diff_image > threshold] = 255

    # Apply mask to thresholded image
    thresholded_image *= mask

    canny_image = cv2.Canny(thresholded_image, 100, 200)

    # Apply close operation to thresholded image
    canny_image = cv2.morphologyEx(canny_image, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)

    all_bounding_boxes = []
    # Loop through patches in the image
    for y in range(0, gray_image.shape[0] - patch_size[0] + 1, stride):
        for x in range(0, gray_image.shape[1] - patch_size[1] + 1, stride):
            patch = canny_image[y:y + patch_size[0], x:x + patch_size[1]]

            # Find contours in the patch
            contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on aspect ratio to distinguish people from other objects
            filtered_contours = [contour for contour in contours if
                                 aspect_ratio_filter(contour, aspect_ratio_threshold)]

            # Convert patch coordinates to the coordinates in the original image
            translated_bounding_boxes = [(x + bx, y + by, bw, bh) for bx, by, bw, bh in
                                         map(cv2.boundingRect, filtered_contours)]

            # Add the bounding boxes to the list
            all_bounding_boxes.extend(translated_bounding_boxes)

    boxes = non_max_suppression(all_bounding_boxes)

    x = np.array([])
    y = np.array([])

    # Loop through all bounding boxes
    for box in boxes:
        # Extract coordinates and dimensions
        x_coord, y_coord, width, height = box

        # Calculate the center coordinates (centroid)
        center_x = x_coord + width // 2
        center_y = y_coord + height // 2

        x = np.append(x, int(center_x))
        y = np.append(y, int(center_y))

    return pd.DataFrame({
        'x': x,
        'y': y
    })

# %%
