import cv2
import numpy as np

img = cv2.imread("scorecards/8bitlair-easy-0.jpg")
h, w = img.shape[:2]
pixels = img.reshape((-1, 3)).astype(np.float32)

# Subsample pixels for faster k-means (e.g. 10,000)
np.random.seed(42)
sample_size = min(10000, len(pixels))
sample_idx = np.random.choice(len(pixels), sample_size, replace=False)
sample_pixels = pixels[sample_idx]

# K-means on sample
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
_, _, centers = cv2.kmeans(sample_pixels, 20, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

# Add fixed color
fixed_color = np.array([[20, 37, 56]], dtype=np.float32)
palette = np.vstack([fixed_color, centers])  # total 10

# Apply to full image
dists = np.linalg.norm(pixels[:, None] - palette[None, :], axis=2)
nearest = np.argmin(dists, axis=1)
quantized = np.round(palette[nearest]).astype(np.uint8).reshape((h, w, 3))

# Save and verify
cv2.imwrite("fast_quantized.jpg", quantized)
unique_colors = np.unique(quantized.reshape(-1, 3), axis=0)
print(f"Unique colors used: {len(unique_colors)}")
