import numpy as np
import cv2
from PIL import Image
from constants import *

import misc


def reorder_contour(contour):
    """Reorders contour points starting from the leftmost, topmost point"""
    # get bounding box
    c = contour.reshape(-1, 1, 2)
    x, y, w, h = cv2.boundingRect(c)

    dist_to_top = np.linalg.norm(contour - (x, y), axis=1)
    contour = np.roll(contour, -np.argmin(dist_to_top), axis=0)

    def is_clockwise(contour):
        """shoelace formula"""
        contour = np.squeeze(contour)
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])  # Shoelace formula
        return area < 0  # Negative area means clockwise

    if is_clockwise(contour):
        contour = contour[::-1]  # Reverse order

    return contour


def resample_contour(contour, num_points=100):
    """Resample a contour to have exactly `num_points` points using interpolation."""
    contour = np.asarray(contour, dtype=np.float32).reshape(-1, 2)  # Ensure shape (N,2)

    # Compute cumulative distances along the contour
    diffs = np.diff(contour, axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))  # Euclidean distances
    cumulative_dists = np.insert(np.cumsum(dists), 0, 0)  # Add 0 at the start

    # Normalize distances to [0,1]
    total_length = cumulative_dists[-1]
    if total_length == 0:
        return contour  # Avoid division by zero for degenerate contours
    normalized_dists = cumulative_dists / total_length

    # Create `num_points` evenly spaced along [0,1]
    target_dists = np.linspace(0, 1, num_points)

    # Interpolate x and y separately
    new_x = np.interp(target_dists, normalized_dists, contour[:, 0])
    new_y = np.interp(target_dists, normalized_dists, contour[:, 1])

    # Combine new points into contour format (Nx1x2 for OpenCV)
    resampled_contour = np.stack((new_x, new_y), axis=1)

    return resampled_contour


def align_contour(contour):
    """Align contours to the origin based on their centroid."""
    centroid = np.mean(contour, axis=0)
    return contour - centroid


def resize_contour(contour):
    """Normalize contour size to fit within a unit bounding box (optional step)."""
    min_x, min_y = np.min(contour, axis=0)
    max_x, max_y = np.max(contour, axis=0)
    scale = max(max_x - min_x, max_y - min_y)  # Keep aspect ratio
    if scale == 0:  # Avoid division by zero
        return contour
    return (contour - [min_x, min_y]) / scale


def standardize_contour(contour, num_points=100):
    contour = reorder_contour(contour)
    contour = resample_contour(contour, num_points)
    contour = align_contour(contour)
    contour = resize_contour(contour)
    return contour


def solidity(contour):
    """Calculate the solidity of a contour. Solidity is the ratio of contour area to its convex hull area."""
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return area / hull_area


def get_contours(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply morphological operations to close gaps in the mask
    kernel = np.ones((5, 5), np.uint8)
    blurred = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    # remove 10% of border (which might have some noise and become a contour)
    h, w = blurred.shape
    border = int(0.1 * min(h, w))
    blurred = blurred[border : h - border, border : w - border]

    def is_border_contour(contour):
        x, y, cw, ch = cv2.boundingRect(contour)
        return x <= 5 or y <= 5 or (x + cw) >= (w - 5) or (y + ch) >= (h - 5)

    # Threshold the image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # save image
    cv2.imwrite("thresh.png", thresh)

    # Find contours for only black objects
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Keep contours that are not border contours and have no parent contour
    # Hierarchy is used to prevent a digit like 8 getting three contours (outer, inner, inner)
    border_idx = -1
    for i, contour in enumerate(contours):
        if is_border_contour(contour):
            border_idx = i
            break
    contours = [c for i,c in enumerate(contours) if not is_border_contour(c) and hierarchy[0][i][3] == border_idx]
    
    
    # remap contour to original size
    for i, contour in enumerate(contours):
        contours[i] = contour + border

    return contours


def get_max_contour(image):
    contours = get_contours(image)
    contours.sort(key=cv2.contourArea, reverse=True)
    if contours is not None:
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    else:
        return None


def get_contours_by_color(image, lower_color, upper_color, required_solidity=None):
    # Create a mask for the color
    mask = cv2.inRange(image, lower_color, upper_color)

    # Apply Gaussian Blur to the mask to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Apply morphological operations to close gaps in the mask
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if required_solidity is not None:
        contours = [c for c in contours if solidity(c) > required_solidity]

    # Find the largest contour
    if contours:
        contours.sort(key=cv2.contourArea, reverse=True)
        return contours
    else:
        return None


def resize_image_to_width(image, width):
    # Calculate the new dimensions while maintaining the aspect ratio
    aspect_ratio = image.shape[0] / image.shape[1]
    new_height = int(width * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_CUBIC)

    return resized_image


def _get_contour_corners(contour):
    # Get the convex hull
    hull = cv2.convexHull(contour)

    # Approximate the contour to reduce number of points
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # If we don't get exactly 4 points, adjust epsilon and try again
    while len(approx) != 4:
        if len(approx) > 4:
            epsilon *= 1.1
        else:
            epsilon *= 0.9
        approx = cv2.approxPolyDP(hull, epsilon, True)

    # Convert to float32 and reshape
    corners = np.float32(approx.reshape(-1, 2))

    # Sort points to ensure consistent order: top-left, top-right, bottom-left, bottom-right
    # Sort by y-coordinate (top to bottom)
    corners = corners[np.argsort(corners[:, 1])]
    # Sort top and bottom pairs by x-coordinate
    top_points = corners[:2][np.argsort(corners[:2, 0])]
    bottom_points = corners[2:][np.argsort(corners[2:, 0])]
    corners = np.vstack([top_points, bottom_points])

    return corners


def downsample_colors(image, num_colors=10, required_colors=None):
    pixels = image.reshape((-1, 3)).astype(np.float32)
    h, w = image.shape[:2]

    sample_size = min(10000, len(pixels))
    sample_idx = np.random.choice(len(pixels), sample_size, replace=False)
    sample_pixels = pixels[sample_idx]

    # K-means on sample
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _, _, centers = cv2.kmeans(sample_pixels, num_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    palette = centers

    # Add fixed color
    if required_colors is not None:
        fixed_color = np.array(required_colors, dtype=np.float32)
        palette = np.vstack([fixed_color, centers])

    # Apply to full image
    dists = np.linalg.norm(pixels[:, None] - palette[None, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    quantized_image = np.round(palette[nearest]).astype(np.uint8).reshape((h, w, 3))

    return quantized_image


def warped_from_contour(image, contour):
    # Find the minimum area rectangle that can enclose the contour
    box = _get_contour_corners(contour)

    # Sort points to ensure consistent order: top-left, top-right, bottom-left, bottom-right
    # Sort by y-coordinate (top to bottom)
    box = box[np.argsort(box[:, 1])]
    top_points = box[:2][np.argsort(box[:2, 0])]
    bottom_points = box[2:][np.argsort(box[2:, 0])]
    box = np.vstack([top_points, bottom_points])

    # 1) get the integer bounding‐rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # 2) build your “destination” quad in the original image’s coordinate frame
    #    in the same order as your `box` ([tl, tr, bl, br]):
    pts_dst_full = np.array(
        [
            [x, y],  # top–left
            [x + w, y],  # top–right
            [x, y + h],  # bottom–left
            [x + w, y + h],  # bottom–right
        ],
        dtype="float32",
    )

    # 3) compute the homography and warp the *whole* image (keep original size)
    M_full = cv2.getPerspectiveTransform(box, pts_dst_full)
    H, W = image.shape[:2]
    full_warped = cv2.warpPerspective(image, M_full, (W, H))

    return full_warped


def extract_image_rects(
    image, contour, color_range=(LOWER_YELLOW_BGR, UPPER_YELLOW_BGR), return_contours=False, save_steps=False
):
    if save_steps:
        misc.plot_contour_on_image(contour, image, output_path="contour_of_rects.png")

    # Find the minimum area rectangle that can enclose the contour
    box = _get_contour_corners(contour)

    # Sort points to ensure consistent order: top-left, top-right, bottom-left, bottom-right
    # Sort by y-coordinate (top to bottom)
    box = box[np.argsort(box[:, 1])]
    top_points = box[:2][np.argsort(box[:2, 0])]
    bottom_points = box[2:][np.argsort(box[2:, 0])]
    box = np.vstack([top_points, bottom_points])

    if save_steps:
        debug_image = image.copy()
        for i, point in enumerate(box):
            cv2.circle(debug_image, tuple(map(int, point)), 5, (0, 0, 255), -1)
            cv2.putText(
                debug_image,
                str(i),
                tuple(map(int, point)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        cv2.imwrite("points.png", debug_image)

    # Calculate width and height
    width_top = np.linalg.norm(box[0] - box[1])
    width_bottom = np.linalg.norm(box[2] - box[3])
    height_left = np.linalg.norm(box[0] - box[2])
    height_right = np.linalg.norm(box[1] - box[3])

    # The width and height of the new image will be the average of the top/bottom and left/right distances
    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))

    pts_dst = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype="float32")

    M = cv2.getPerspectiveTransform(box, pts_dst)
    warped_image = cv2.warpPerspective(image, M, (width, height))

    warped_image = resize_image_to_width(warped_image, 5000)
    if save_steps:
        cv2.imwrite("warped.png", warped_image)

    rects = extract_yellow_rectangles(
        warped_image,
        color_range[0],
        color_range[1],
        return_contours=return_contours,
        save_images=save_steps,
    )

    return warped_image, rects


def get_color_range(color, pct):
    val = pct * 255
    lower = np.array(
        [
            min(255, color[0] - val),
            min(255, color[1] - val),
            min(255, color[2] - val),
        ],
        dtype=np.uint8,
    )
    upper = np.array(
        [
            min(255, color[0] + val),
            min(255, color[1] + val),
            min(255, color[2] + val),
        ],
        dtype=np.uint8,
    )
    return lower, upper


def get_score_section(file_or_bytes, return_rect_contours=False, save_steps=False):
    # Reads BGR image from file or bytes
    if isinstance(file_or_bytes, bytes):
        image = cv2.imdecode(np.frombuffer(file_or_bytes, np.uint8), cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(file_or_bytes)
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, borderType=cv2.BORDER_CONSTANT, value=0)  # helps with contour

    rects = None
    score_image = None
    course_image = None
    for i in np.arange(0.02, 0.1, 0.01):
        lower_color, upper_color = get_color_range(BROWN_BGR, i)
        contours = get_contours_by_color(image, lower_color, upper_color, required_solidity=SCORECARD_SOLIDITY)
        if contours is None:
            continue
        contours = [c for c in contours if cv2.contourArea(c) > MIN_SCORECARD_AREA * image.shape[0] * image.shape[1]]

        for contour in contours:
            for j in np.arange(0.2, 0.41, 0.1):
                color_range = get_color_range(YELLOW_BGR, j)
                score_image, rects = extract_image_rects(
                    image, contour, color_range, return_contours=return_rect_contours, save_steps=save_steps
                )
                if len(rects) > 0 and len(rects) % 18 == 0:
                    break
            if len(rects) > 0 and len(rects) % 18 == 0:
                # Show contour on image
                if save_steps:
                    misc.plot_contour_on_image(contour, image, output_path="contour_scores.png")
                break
            else:
                rects = None

        if rects is not None:
            break

    if rects is not None and score_image is not None:
        x, y, w, h = cv2.boundingRect(contour)
        warped = warped_from_contour(image, contour)
        warped = warped[max(0, y - w // 8) : y + h, x : x + w]
        warped = resize_image_to_width(warped, 5000)
        course_image = warped[50:450, 1000:]

    return course_image, score_image, rects


def extract_yellow_rectangles(image, lower_color, upper_color, return_contours=False, save_images=False):
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply morphological operations to close gaps in the mask
    kernel = np.ones((5, 5), np.uint8)
    blurred = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    if save_images:
        cv2.imwrite("blurred_image.png", blurred)

    # Create a mask for the yellow color
    mask = cv2.inRange(blurred, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # Remove noise (checkmarks)
    if save_images:
        cv2.imwrite("mask_image.png", mask)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def valid_contour(c):
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h
        return 0.75 <= aspect_ratio <= 1.2 and w > 100 and w < 200

    contours = [c for c in contours if valid_contour(c)]

    # Get each row of score contours
    # Base this off similar (but not exact) y positions
    contours.sort(key=lambda x: cv2.boundingRect(x)[1])  # sort by y position
    contour_rows = []
    prev_y = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if len(contour_rows) == 0:
            contour_rows.append([c])
            prev_y = y
        else:
            if y - prev_y < 0.25 * h:
                contour_rows[-1].append(c)
            else:
                contour_rows.append([c])
            prev_y = y

    # Sort by x values
    for i in range(len(contour_rows)):
        contour_rows[i].sort(key=lambda x: cv2.boundingRect(x)[0])

    contours = [c for row in contour_rows for c in row]
    if return_contours:
        return contours

    rectangle_images = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = image[y : y + h, x : x + w]
        pil_image = Image.fromarray(cropped_image)
        rectangle_images.append(pil_image)
        if save_images:
            pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            pil_image.save(f"rectangles/rect_{i}.png")

    return rectangle_images


def get_par_contours(image_path):
    course_image, score_image, rects = get_score_section(image_path, save_steps=False)

    # Get area of image containing par values
    x1, y1, _, _ = cv2.boundingRect(rects[0])
    x2, y2, w2, _ = cv2.boundingRect(rects[17])
    y = min(y1, y2) - 10
    image = score_image[y - 100 : y, x1 : x2 + w2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours for only black objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    def height(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return h

    contours = [c for c in contours if height(c) > 50]
    contours.sort(key=lambda x: min(x[:, 0, 0]))

    return contours


def digits_from_score_rect(rect):
    rect = np.asarray(rect).copy()
    contours = get_contours(rect)
    digit_contours = [c.reshape(-1, 2) for c in contours]
    return digit_contours
