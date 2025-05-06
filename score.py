import numpy as np
import argparse

from misc import *
from scorecard import *
from image_manipulation import *

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
        s = Scorecard.from_image(image_path, standard_contours)
        scorecard = s.copy() if scorecard is None else scorecard.combine(s)
    if scorecard is None:
        print("No scorecard found.")
        return

    scorecard.include_best = True
    scorecard.include_pars = True
    pardiff = scorecard.compare_to_par()
    bestdiff = scorecard.compare_to_best()
    print(df_to_str(scorecard.summarize_shots()))
    print(df_to_str(scorecard.summarize_scores()))
    dfs_to_image(
        [scorecard.df, pardiff.df, bestdiff.df],
        titles=["Scorecard", "Par Diff", "Best Diff"],
        output_path="__tmp_tables.png",
    )


if __name__ == "__main__":
    main()
