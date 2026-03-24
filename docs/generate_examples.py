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
from utils import load_standard_contours, dfs_to_image, render_templates_grid
from scorecard import Scorecard, get_player_region

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = "data/scorecards/quixotevalley-hard-0.jpg"

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


def save(name, img):
    cv2.imwrite(os.path.join(DOCS_DIR, name), img)
    print(name)


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULTS
    if cv2.imread(image_path) is None:
        print(f"Error: could not read {image_path}")
        sys.exit(1)

    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs("rectangles", exist_ok=True)

    # Pipeline with save_steps generates debug images (detected, warped, mask)
    course_image, _, rects = get_score_section(image_path, save_steps=True)
    for src, dest in {
        "contour_scores.png": "detected.png",
        "warped.png": "warped.png",
        "mask_image.png": "mask.png",
    }.items():
        try:
            shutil.move(src, os.path.join(DOCS_DIR, dest))
            print(dest)
        except FileNotFoundError:
            print(f"  (skipped {src})")

    # Course name region
    if course_image is not None:
        save("course.png", course_image)

    # Player name region
    _, score_image, rect_contours = get_score_section(image_path, return_rect_contours=True)
    if rect_contours and score_image is not None:
        player_region = get_player_region(score_image, rect_contours)
        ph, pw = player_region.shape[:2]
        if ph > 0:
            scale = 300 / ph
            save("players.png", cv2.resize(player_region, (int(pw * scale), int(ph * scale))))

    # Digit recognition: raw rect, contour overlay, standardized contour
    if rects:
        rect_arr = np.asarray(rects[0])
        save("digit_rect.png", cv2.resize(rect_arr, (200, 200), interpolation=cv2.INTER_NEAREST))

        contours = digits_from_score_rect(rects[0])
        if contours:
            rect_outlined = np.asarray(rects[0]).copy()
            for c in contours:
                cv2.drawContours(rect_outlined, [c.reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
            save("digit_outlined.png", cv2.resize(rect_outlined, (200, 200), interpolation=cv2.INTER_NEAREST))

            sc = standardize_contour(contours[0])
            sc_img = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.drawContours(sc_img, [(sc.reshape(-1, 1, 2) * 250 + 25).astype(np.int32)], -1, (0, 255, 0), 2)
            save("contour.png", sc_img)

    # Standard contour templates grid (reuses render_templates_grid from utils.py)
    standard_contours = load_standard_contours()
    save("templates.png", render_templates_grid(standard_contours))

    # Full output tables
    scorecard = Scorecard.from_image(image_path, standard_contours)
    scorecard.include_pars = True
    scorecard.include_best = True
    dfs_to_image(
        [
            scorecard.summarize_scores(),
            scorecard.summarize_shots(),
            scorecard.df,
            scorecard.compare_to_par().df,
            scorecard.compare_to_best().df,
        ],
        titles=["Summary", "Shots", "Scorecard", "Par Diff", "Best Diff"],
        output_path=os.path.join(DOCS_DIR, "output.png"),
    )
    print("output.png")

    # Cleanup debug images
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
