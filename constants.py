import numpy as np
import cv2

IMG_TYPES = ("jpg", "webp", "png")
SCORECARD_SOLIDITY = 0.50
MIN_SCORECARD_AREA = 0.01

YELLOW_BGR = np.array([100, 255, 255], dtype=np.uint8)
BROWN_BGR = np.array([20, 37, 56], dtype=np.uint8)
_YELLOW_RANGE = 0.2 * 255
_BROWN_RANGE = 0.05 * 255

LOWER_YELLOW_BGR = np.array(
    [
        max(0, YELLOW_BGR[0] - _YELLOW_RANGE),
        max(0, YELLOW_BGR[1] - _YELLOW_RANGE),
        max(0, YELLOW_BGR[2] - _YELLOW_RANGE),
    ],
    dtype=np.uint8,
)
UPPER_YELLOW_BGR = np.array(
    [
        min(255, YELLOW_BGR[0] + _YELLOW_RANGE),
        min(255, YELLOW_BGR[1] + _YELLOW_RANGE),
        min(255, YELLOW_BGR[2] + _YELLOW_RANGE),
    ],
    dtype=np.uint8,
)

LOWER_BROWN_BGR = np.array(
    [max(0, BROWN_BGR[0] - _BROWN_RANGE), max(0, BROWN_BGR[1] - _BROWN_RANGE), max(0, BROWN_BGR[2] - _BROWN_RANGE)],
    dtype=np.uint8,
)
UPPER_BROWN_BGR = np.array(
    [
        min(255, BROWN_BGR[0] + _BROWN_RANGE),
        min(255, BROWN_BGR[1] + _BROWN_RANGE),
        min(255, BROWN_BGR[2] + _BROWN_RANGE),
    ],
    dtype=np.uint8,
)
