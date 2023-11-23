import cv2


def dilate(image, kernel):
    return cv2.dilate(kernel)


def erode(image, kernel):
    return cv2.erode(kernel)

def opening(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def closing(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
