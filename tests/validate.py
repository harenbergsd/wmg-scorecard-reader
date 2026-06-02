import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from utils import load_standard_contours
from scorecard import Scorecard
from constants import IMG_TYPES

directory = Path("data/scorecards")
image_files = [file for ext in IMG_TYPES for file in directory.glob(f"*.{ext}")]

for image_file in image_files:
    print(f"Comparing {image_file.name}...", end=" ")

    solution_file = image_file.with_suffix(".sol")
    gt = Scorecard.from_solution_file(solution_file)

    standard_contours = load_standard_contours()
    scorecard = Scorecard.from_image(image_file, standard_contours)
    if gt != scorecard:
        print(f"✗")
        df = gt.df.reset_index(drop=True)
        scorecard_df = scorecard.df.reset_index(drop=True)
        print(scorecard.course)
        print(gt.course)
        print(df)
        print(scorecard_df)
        break
    else:
        print(f"✓")
