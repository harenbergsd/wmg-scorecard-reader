"""Generate example pipeline images for the README.

Usage:
    python docs/generate_examples.py [image_path]

Defaults to data/scorecards/quixotevalley-hard-0.jpg if no argument given.
"""
import sys
import os
import shutil
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_manipulation import get_score_section, digits_from_score_rect, standardize_contour
from utils import load_standard_contours, dfs_to_image
from scorecard import Scorecard

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = "data/scorecards/quixotevalley-hard-0.jpg"

# Max widths for downsizing
MAX_WIDTHS = {
    "detected.png": 800,
    "warped.png": 1000,
    "mask.png": 1000,
    "course.png": 800,
}


def downsize(path, max_width):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        cv2.imwrite(path, img)


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULTS
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not read {image_path}")
        sys.exit(1)

    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs("rectangles", exist_ok=True)

    # 1. Run pipeline with save_steps for intermediate images
    course_image, score_image, rects = get_score_section(image_path, save_steps=True)

    moves = {
        "contour_scores.png": "detected.png",
        "warped.png": "warped.png",
        "mask_image.png": "mask.png",
    }
    for src, dest in moves.items():
        dest_path = os.path.join(DOCS_DIR, dest)
        try:
            shutil.move(src, dest_path)
            print(dest)
        except FileNotFoundError:
            print(f"  (skipped {src} — not generated)")

    # 2. Course image
    if course_image is not None:
        cv2.imwrite(os.path.join(DOCS_DIR, "course.png"), course_image)
        print("course.png")

    # 3. Single score rectangle
    if rects:
        rect_arr = np.asarray(rects[0])
        rect_big = cv2.resize(rect_arr, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(DOCS_DIR, "digit_rect.png"), rect_big)
        print("digit_rect.png")

    # 4. Standardized digit contour
    if rects:
        contours = digits_from_score_rect(rects[0])
        if contours:
            sc = standardize_contour(contours[0])
            sc_img = np.zeros((300, 300, 3), dtype=np.uint8)
            sc_draw = (sc.reshape(-1, 1, 2) * 250 + 25).astype(np.int32)
            cv2.drawContours(sc_img, [sc_draw], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(DOCS_DIR, "contour.png"), sc_img)
            print("contour.png")

    # 5. Full output tables
    standard_contours = load_standard_contours()
    scorecard = Scorecard.from_image(image_path, standard_contours)
    scorecard.include_pars = True
    scorecard.include_best = True
    summary = scorecard.summarize_scores()
    shots = scorecard.summarize_shots()
    pardiff = scorecard.compare_to_par()
    bestdiff = scorecard.compare_to_best()
    dfs_to_image(
        [summary, shots, scorecard.df, pardiff.df, bestdiff.df],
        titles=["Summary", "Shots", "Scorecard", "Par Diff", "Best Diff"],
        output_path=os.path.join(DOCS_DIR, "output.png"),
    )
    print("output.png")

    # Cleanup leftover debug images
    for f in ["contour_of_rects.png", "points.png", "blurred_image.png"]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    # Downsize large images
    for name, max_w in MAX_WIDTHS.items():
        path = os.path.join(DOCS_DIR, name)
        if os.path.exists(path):
            downsize(path, max_w)

    print(f"\nDone — images saved to {DOCS_DIR}/")


if __name__ == "__main__":
    main()
