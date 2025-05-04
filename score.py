import numpy as np
import argparse

from misc import *
from scorecard import *
from image_manipulation import *

argparser = argparse.ArgumentParser(description="Get the score from the scorecard image")
argparser.add_argument("image_path", help="Path to the image file")
argparser.add_argument(
    "--standard_contours",
    default="standard_contours.pkl",
    help="Path to the standard contours file",
)
args = argparser.parse_args()


def main():
    standard_contours = load_standard_contours()
    scorecard = Scorecard.from_image(args.image_path, standard_contours)
    scorecard.include_best = True
    scorecard.include_pars = True
    pardiff = scorecard.compare_to_par()
    bestdiff = scorecard.compare_to_best()
    print(scorecard.course)
    print(scorecard.summarize_shots())
    print(scorecard.summarize_scores())
    dfs_to_image(
        [scorecard.df, pardiff.df, bestdiff.df],
        titles=["Scorecard", "Par Diff", "Best Diff"],
        output_path="__tmp_tables.png",
    )


if __name__ == "__main__":
    main()
