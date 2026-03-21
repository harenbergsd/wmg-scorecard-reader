import numpy as np
import cv2

IMG_TYPES = ("jpg", "webp", "png")
SCORECARD_SOLIDITY = 0.50
MIN_SCORECARD_AREA = 0.01

YELLOW_BGR = np.array([100, 255, 255], dtype=np.uint8)
BROWN_BGR = np.array([20, 37, 56], dtype=np.uint8)

LOWER_YELLOW_BGR = np.array(
    [max(0, YELLOW_BGR[0] - 0.2 * 255), max(0, YELLOW_BGR[1] - 0.2 * 255), max(0, YELLOW_BGR[2] - 0.2 * 255)],
    dtype=np.uint8,
)
UPPER_YELLOW_BGR = np.array(
    [min(255, YELLOW_BGR[0] + 0.2 * 255), min(255, YELLOW_BGR[1] + 0.2 * 255), min(255, YELLOW_BGR[2] + 0.2 * 255)],
    dtype=np.uint8,
)
