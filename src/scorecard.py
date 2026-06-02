import logging
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageEnhance
from scipy.spatial import procrustes

from image_manipulation import get_score_section, digits_from_score_rect, standardize_contour
import utils
import warnings

warnings.filterwarnings("ignore", message="No ccache found.*", category=UserWarning)

from paddleocr import PaddleOCR

logging.getLogger("paddle").setLevel(logging.WARNING)  # Suppresses PaddlePaddle logs
logging.getLogger("ppocr").setLevel(logging.WARNING)  # Suppresses PaddleOCR logs

PAR_NAME = "<PAR>"
BEST_NAME = "<BEST>"


class Scorecard:
    def __init__(self, course, players, scores, pars_csv=None):
        self._course = course
        self._players = list(players)
        self._scores = list(scores)
        self._include_pars = False
        self._include_best = False
        self._pars = None

        # remove players with all NaN scores
        indexes = []
        for i in range(len(scores)):
            if all(np.isnan(scores[i])):
                indexes.append(i)
        self._players = [p for i, p in enumerate(players) if i not in indexes]
        self._scores = [s for i, s in enumerate(scores) if i not in indexes]
        if len(players) == 0:
            raise ValueError("No players with scores")

        self._df = pd.DataFrame(self._scores, columns=list(range(1, 19)), index=pd.Index(self._players))
        self._df.index.name = self.course
        mask_all_na = self._df.notna().any(axis=1)
        self._df["total"] = self._df.sum(axis=1, skipna=True).mask(~mask_all_na)
        self._df = self._df.astype("Int64")
        self._set_pars()

    def sorted_by_total(self):
        s = self.copy()
        s._df = s._df.sort_values(by=["total", s._df.index.name or s._df.index])
        s._players = [p for p in s._df.index.tolist() if p in s._players]
        return s

    def combine(self, other):
        if self.course != other.course:
            raise ValueError("Courses do not match")
        s = self.copy()
        o = other.copy()
        o.include_pars = False
        o.include_best = False
        s._df = pd.concat([s._df, o.df])
        s._players += o.players
        s._scores += o.scores
        if s.include_best:
            # recalculate best scores
            s.include_best = False
            s.include_best = True
        return s

    def compare_to_best(self):
        s = self.copy()
        s.include_best = True
        s._df[s._df["total"] == 0] = 99
        s = s.sorted_by_total()
        s._df[s._df["total"] == 99] = np.nan
        s._df = s._df - s._df.loc[BEST_NAME]
        s._df = s._df.drop(BEST_NAME)
        return s

    def compare_to_par(self):
        if self._pars is None:
            raise ValueError("No pars available")
        s = self.copy()
        s.include_pars = True
        s._df = s._df.drop(PAR_NAME)
        s._df = s._df - s.pars
        # recompute total since pars is 18 holes and scorecard might be 9 holes
        s._df = s._df.drop(columns="total")
        mask_all_na = s._df.notna().any(axis=1)
        s._df["total"] = s._df.sum(axis=1, skipna=True).mask(~mask_all_na)
        s._df = s._df.astype("Int64")
        return s

    def _set_pars(self, pars_csv="data/pars.csv"):
        pars = pd.read_csv(pars_csv, index_col="course")

        # calculate string similarity for each course
        pars["similarity"] = pars.index.map(lambda x: utils.string_edit_distance(x, self.course))

        best_match_idx = pars["similarity"].idxmin()
        pars = pars.drop(columns=["similarity", "code"])
        self._pars = pars.loc[best_match_idx].values

    def summarize_scores(self):
        """Computers the front 9 (in), back 9 (out), and total scores for each player"""
        s = self.copy()
        s.include_best = False
        scores = s.compare_to_par().df
        scores = scores.drop(columns="total")
        front_nine = scores.iloc[:, :9].sum(axis=1)
        back_nine = scores.iloc[:, 9:].sum(axis=1)
        total = front_nine + back_nine

        result = pd.DataFrame(
            {
                "in": front_nine,
                "out": back_nine,
                "total": total,
            },
            index=s.players,
        )
        result.index.name = self.course

        return result

    def summarize_shots(self):
        """count the scores compared to par per player"""
        s = self.copy()
        scores = s.compare_to_par()
        s.include_pars = False
        s.include_best = False
        scores.include_best = False
        scores = scores.df

        scores = scores.drop(columns="total")
        best = scores.min().min()
        worst = scores.max().max()

        result = pd.DataFrame(index=s.players, columns=range(best, worst + 1))
        result.index.name = self.course

        for i in range(best, worst + 1):
            counts = scores.applymap(lambda x: 1 if pd.notna(x) and x == i else 0).sum(axis=1)
            result.loc[:, i] = counts.to_numpy()

        # get number of hole in one scores
        result.loc[:, "hi1s"] = s.df.applymap(lambda x: 1 if pd.notna(x) and x == 1 else 0).sum(axis=1).values

        return result

    @property
    def course(self):
        return self._course

    @property
    def players(self):
        return self._players

    @property
    def scores(self):
        return self._scores

    @property
    def df(self):
        return self._df

    @property
    def pars(self):
        return self._pars

    @property
    def include_pars(self):
        return self._include_pars

    @include_pars.setter
    def include_pars(self, value):
        if value and self._pars is None:
            raise ValueError("No pars available. Add them with the add_pars method.")
        self._include_pars = value
        if not value:
            self._df = self._df.drop(PAR_NAME, errors="ignore")
        if value and PAR_NAME not in self._df.index:
            pars = pd.Series(self._pars, name=PAR_NAME, index=self.df.columns)
            self._df = pd.concat([pars.to_frame().T, self._df])
            self._df.index.name = self.course

    @property
    def include_best(self):
        return self._include_best

    @include_best.setter
    def include_best(self, value):
        self._include_best = value
        if not value:
            self._df = self._df.drop(BEST_NAME, errors="ignore")
        if value and BEST_NAME not in self._df.index:
            df = self._df.drop(PAR_NAME, errors="ignore")
            best = pd.Series(df.min(), name=BEST_NAME, index=self.df.columns)
            self._df = pd.concat([best.to_frame().T, self._df])
            self._df.loc[BEST_NAME, "total"] = 0
            self._df.loc[BEST_NAME, "total"] = self._df.loc[BEST_NAME].sum()
            self._df.index.name = self.course

    def __str__(self):
        df = self.df.astype(object).fillna("-")
        return f"{self.course}\n{df.to_markdown(tablefmt='grid')}"

    def __repr__(self):
        return f"Scorecard(course={self.course}, players={self.players}, scores={self.scores})"

    def __eq__(self, other):
        # check if the course, players, and scores are the same
        # compare dataframes
        tmp_self = self.df.reset_index(drop=True)
        tmp_other = other.df.reset_index(drop=True)
        return tmp_self.equals(tmp_other) and self.course == other.course

    def copy(self):
        s = Scorecard(self.course, self.players, self.scores)
        s._df = self.df.copy()
        s._pars = self.pars
        s._include_pars = self.include_pars
        s._include_best = self.include_best
        return s

    @classmethod
    def from_solution_file(cls, solution_file):
        with open(solution_file, "r") as f:
            lines = f.readlines()
        course = lines[0].strip()
        lines = lines[1:]
        players = [line.split(",")[0] for line in lines]
        # create self.scores with int or nan
        scores = []
        for line in lines:
            s = line.strip().split(",")[1:]
            scores.append([np.nan if x == "-" else int(x) for x in s])

        return cls(course, players, scores)

    @classmethod
    def from_image(cls, image_file, standard_contours):
        course = get_course_from_image(image_file)
        players = get_players_from_image(image_file)
        if not players:
            raise ValueError(f"Could not read player names from scorecard (detected course: {course})")
        scores = get_scores_from_image(image_file, standard_contours)
        if not scores:
            raise ValueError(f"Could not read scores from scorecard (detected course: {course})")
        scores = [scores[i : i + 18] for i in range(0, len(scores), 18)]
        return cls(course, players, scores)


def get_course_from_image(image_path, min_confidence=0.5):
    course_image, _, _ = get_score_section(image_path, save_steps=False)
    if course_image is None:
        raise ValueError(
            "Failed to recognize the structure of the scorecard. Make sure the image is correct and clear."
        )

    ocr = PaddleOCR(lang="en")
    results = ocr.ocr(np.array(course_image), cls=False)
    if not results or results[0] is None:
        raise ValueError(
            "Failed to recognize the structure of the scorecard. Make sure the image is correct and clear."
        )
    full_results = [box for box in results[0]]

    # Keep only the top row of text (course name line).
    # The status line (MODE, RECORD, ROOM, etc.) is on a lower row.
    # Find the bottom edge of the topmost box, then keep only boxes
    # whose top edge is within that range.
    top_box_bottom = min(max(p[1] for p in box[0]) for box in full_results)
    results = [box for box in full_results if min(p[1] for p in box[0]) <= top_box_bottom]

    # ocr_result: List of [box, (text, confidence)]
    # order by x coordinate
    results = sorted(results, key=lambda x: x[0][0][0])

    # Gather all detected text, filtering out known UI words and status items
    words = []
    ignore_words = {"mode:", "record:", "room:"}
    for box in results:
        t, conf = box[1]
        t_lower = t.strip().lower()
        if conf > min_confidence and t_lower not in ignore_words:
            words.append(t)
    text = " ".join(words)

    # Detect unsupported race mode scorecards
    results = [box for box in full_results if min(p[1] for p in box[0]) > top_box_bottom]
    for box in results:
        t, _ = box[1]
        if "mode:" in t.strip().lower() and "race" in t.strip().lower():
            raise ValueError("Race mode scorecards are not supported (times instead of scores).")

    # load courses from pars.csv
    pars = pd.read_csv("data/pars.csv", index_col="course")
    courses = [c for c in pars.index.tolist()]

    return min(courses, key=lambda c: utils.string_edit_distance(text, c))


def get_player_region(score_image, rect_contours):
    """Crop the player name region from the left side of the warped score image."""
    nrows = len(rect_contours) // 18
    x2, y1, _, _ = cv2.boundingRect(rect_contours[0])
    _, y2, _, h = cv2.boundingRect(rect_contours[(nrows - 1) * 18])
    return score_image[y1 : (y2 + h), 0:x2]


def get_players_from_image(image_path):
    _, score_image, rects = get_score_section(image_path, return_rect_contours=True, save_steps=False)
    if rects is None:
        raise ValueError("Could not find the score grid in the image")

    image = get_player_region(score_image, rects)

    # improve image quality
    image = Image.fromarray(image)
    image = image.convert("L")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Use PaddleOCR
    ocr = PaddleOCR(lang="en")
    results = ocr.ocr(np.array(image), cls=False)
    if not results or results[0] is None:
        raise ValueError("Could not read player names from image")
    players = []
    for i, line in enumerate(results[0]):
        text, confidence = line[1]
        players.append(text.upper())

    return players


def get_scores_from_image(image_path, standard_contours):
    scores = []
    _, _, rects = get_score_section(image_path, save_steps=False)
    if rects is None:
        raise ValueError("Could not find the score grid in the image")
    for r in rects:
        digit_contours = digits_from_score_rect(r)
        scorestr = ""
        for d in digit_contours:
            scorestr += str(match_digit_contour(d, standard_contours))
        scores.append(int(scorestr) if scorestr.isdigit() else np.nan)
    return scores


def best_contour_alignment(c1, c2):
    """Finds the best circular shift for contour c2 to match c1."""
    best_shift = 0
    best_distance = float("inf")
    best_c2 = c2

    # Try subset of shifts left and right
    # Don't want to use all shifts because 6 and 9 are the same (although I don't think there are any 9s)
    for shift in range(-10, 10):
        shifted_c2 = np.roll(c2, shift, axis=0)
        distance = np.sum(np.linalg.norm(c1 - shifted_c2, axis=1))  # Fast Euclidean sum

        if distance < best_distance:
            best_distance = distance
            best_shift = shift
            best_c2 = shifted_c2

    return best_c2, best_shift, best_distance


def match_digit_contour(digit_contour, standard_contours):
    num_points = list(standard_contours.values())[0].shape[0]
    digit_contour = standardize_contour(digit_contour, num_points)

    min_dist = float("inf")
    best_match = None
    for digit, standard_contour in standard_contours.items():
        c, _, _ = best_contour_alignment(standard_contour, digit_contour)
        _, _, dist = procrustes(c, standard_contour)
        if dist < min_dist:
            min_dist = dist
            best_match = digit

    return best_match
