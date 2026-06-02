import sys
import os
import numpy as np
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils import load_standard_contours, dfs_to_image
from scorecard import Scorecard

argparser = argparse.ArgumentParser(description="Get the score from the scorecard image")
# get possibly multiple paths to images
argparser.add_argument(
    "image_path",
    type=str,
    nargs="+",
    help="Path to the scorecard image(s). Can be a single image or multiple images.",
)
args = argparser.parse_args()


def main():
    standard_contours = load_standard_contours()
    scorecard = None
    for image_path in args.image_path:
        try:
            s = Scorecard.from_image(image_path, standard_contours)
        except ValueError as e:
            print(f"Skipping {image_path}: {e}")
            continue
        scorecard = s.copy() if scorecard is None else scorecard.combine(s)
    if scorecard is None:
        print("No scorecard found.")
        return

    scorecard.include_best = True
    scorecard.include_pars = True
    summary = scorecard.summarize_scores()
    shots = scorecard.summarize_shots()
    pardiff = scorecard.compare_to_par()
    bestdiff = scorecard.compare_to_best()
    print(scorecard)
    dfs_to_image(
        [summary, shots, scorecard.df, pardiff.df, bestdiff.df],
        titles=["Summary", "Shots", "Scorecard", "Par Diff", "Best Diff"],
    )


if __name__ == "__main__":
    main()
