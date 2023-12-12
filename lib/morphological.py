import cv2


def opening(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def closing(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
