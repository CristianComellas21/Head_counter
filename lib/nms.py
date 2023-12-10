import cv2
import numpy as np


def aspect_ratio_filter(contour, aspect_ratio_threshold):
    # Calculate aspect ratio of a contour
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)

    # Filter contours based on aspect ratio
    return aspect_ratio >= aspect_ratio_threshold


def non_max_suppression(boxes, threshold=0.5):
    """
    Apply non-maximum suppression to eliminate redundant bounding boxes.

    Args:
    - boxes (list): List of bounding boxes in the format (x, y, w, h).
    - threshold (float): Intersection over Union (IoU) threshold for suppression.

    Returns:
    - List of selected bounding boxes after NMS.
    """
    if len(boxes) == 0:
        return []

    eps = 0.0001
    # Convert to NumPy array for easier calculations
    boxes = np.array(boxes)

    # Extract coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    # Sort bounding boxes by their area (smaller boxes first)
    area = boxes[:, 2] * boxes[:, 3]
    indices = np.argsort(area)

    # Initialize list to store selected bounding boxes
    selected_boxes = []

    while len(indices) > 0:
        # Select the bounding box with the smallest area
        current_index = indices[0]
        selected_boxes.append(current_index)

        # Calculate intersection over union (IoU) with other bounding boxes
        intersection_x1 = np.maximum(x1[current_index], x1[indices[1:]])
        intersection_y1 = np.maximum(y1[current_index], y1[indices[1:]])
        intersection_x2 = np.minimum(x2[current_index], x2[indices[1:]])
        intersection_y2 = np.minimum(y2[current_index], y2[indices[1:]])

        intersection_width = np.maximum(0, intersection_x2 - intersection_x1 + 1)
        intersection_height = np.maximum(0, intersection_y2 - intersection_y1 + 1)

        intersection_area = intersection_width * intersection_height

        # Calculate union and IoU
        union_area = (area[current_index] + area[indices[1:]] - intersection_area) + eps
        iou = intersection_area / union_area

        # Remove bounding boxes with IoU greater than the threshold
        suppressed_indices = np.where(iou <= threshold)[0]
        indices = indices[suppressed_indices + 1]

    # Return the selected bounding boxes
    return boxes[selected_boxes]
