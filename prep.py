import numpy as np
import pickle
import shutil
import os
import argparse
from collections import defaultdict
from scipy.spatial import procrustes

from image_manipulation import *
from constants import *
from scorecard import *


argparser = argparse.ArgumentParser(description="Extract scores from scorecard image for training")
argparser.add_argument("image_path", help="Path to the image file or folder containing images")
argparser.add_argument(
    "--save_score_rectangles",
    help="Save the extracted score rectangles to a file",
    action="store_true",
)
argparser.add_argument(
    "--save_contours",
    help="Save the extracted digit contours to a file",
    action="store_true",
)

args = argparser.parse_args()


def save_score_rectangles(rects):
    # clear the rectangles folder
    for f in os.listdir("rectangles"):
        os.remove(os.path.join("rectangles", f))
    # save rectangle images
    for i, rect in enumerate(rects):
        cv2.imwrite(f"rectangles/rect_{i}.png", rect)


def get_digit_templates():
    for f in os.listdir("data/scorecards"):
        os.remove(os.path.join("digits", f))


def test_digit_extraction(rect_image_path):
    rect_image = cv2.imread(rect_image_path)
    rect = np.asarray(rect_image)
    digit_contour = get_max_contour(rect)

    if digit_contour is None or len(digit_contour) == 0:
        raise ValueError("Error: No digit contours found in the specified color range.")

    # draw the contour
    cv2.drawContours(rect, digit_contour, -1, (0, 255, 0), 3)
    cv2.imwrite("digit_contour.png", rect)


def compute_average_contour(contours, num_points=100):
    """Compute the average contour from multiple contours."""
    contours = [reorder_contour(c) for c in contours]
    contours = [resample_contour(c, num_points) for c in contours]
    contours = [align_contour(c) for c in contours]
    contours = [resize_contour(c) for c in contours]

    mean_contour = np.mean(contours, axis=0)

    return mean_contour


def get_training_data():
    image_path = args.image_path
    if os.path.isdir(image_path):
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.lower().endswith(IMG_TYPES)]
    else:
        image_files = [image_path]

    score_contours = defaultdict(list)
    for image_file in image_files:
        print(image_file)
        contour_dirname = os.path.join("data/contours", os.path.basename(image_file).split(".")[0])

        # Read the solution file
        solution_filepath = image_file.split(".")[0] + ".sol"
        solution = Scorecard.from_solution_file(solution_filepath)

        # Extract the score contour from the image
        scores = [s for player_scores in solution.scores for s in player_scores]
        digits = [int(d) for score in scores for d in str(score) if score is not np.nan]
        digits += [np.nan] if np.nan in scores else []
        contours = {i: [] for i in set(digits)}

        if args.save_contours:
            shutil.rmtree(contour_dirname, ignore_errors=True)
            for score in contours.keys():
                os.makedirs(os.path.join(contour_dirname, str(score)), exist_ok=True)

        _, _, rects = get_score_section(image_file, save_steps=False)
        for i, rect in enumerate(rects):
            rect = np.asarray(rect).copy()
            digit_contours = get_contours(rect)
            real_digits = [scores[i]]
            if scores[i] is not np.nan:
                real_digits = [
                    int(d) for d in str(scores[i])
                ]  # double digit scores are possible, e.g. 10, 11, 12, etc.
            for real_digit, contour in zip(real_digits, digit_contours):
                if args.save_contours:
                    score = scores[i]
                    cv2.drawContours(rect, contour, -1, (0, 255, 0), 3)
                    cv2.imwrite(os.path.join(contour_dirname, str(score), f"rect_{i}.png"), rect)

                contours[real_digit].append(contour.reshape(-1, 2))

        for score, digit_contours in sorted(contours.items()):
            score_contours[score] += digit_contours

    for score, contours in sorted(score_contours.items()):
        print(score, len(contours))
    print(sum([len(contours) for contours in score_contours.values()]))

    if args.save_contours:
        with open("score_contours.pkl", "wb") as f:
            pickle.dump(score_contours, f)


def main():
    get_training_data()

    # read pickle file
    with open("score_contours.pkl", "rb") as f:
        score_contours = pickle.load(f)

    # compute average contour
    standard_contours = {}
    for score, contours in score_contours.items():
        mean_contour = compute_average_contour(contours)
        standard_contours[score] = mean_contour

        # save the mean contour image
        if args.save_contours:
            mean_contour = mean_contour.reshape(-1, 1, 2) * 1000
            mean_contour = mean_contour.astype(np.int32)
            img = np.zeros((1000, 1000, 3), dtype=np.uint8)
            cv2.drawContours(img, [mean_contour], contourIdx=-1, color=(255, 255, 255), thickness=2)
            cv2.imwrite(f"{score}.png", img)

    # save the standard contours
    with open("data/standard_contours.pkl", "wb") as f:
        pickle.dump(standard_contours, f)


if __name__ == "__main__":
    main()
