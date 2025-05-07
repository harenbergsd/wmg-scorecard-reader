import cv2
import numpy as np
import pickle
from table2ascii import table2ascii as t2a, PresetStyle, Alignment


def plot_contour(c, output_path="contour.png"):
    c = c.reshape(-1, 1, 2) * 1000
    c = c.astype(np.int32)
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cv2.drawContours(img, [c], contourIdx=-1, color=(255, 255, 255), thickness=2)
    cv2.imwrite(output_path, img)


def plot_contour_on_image(c, img, output_path="contour.png"):
    contour_image = img.copy()
    cv2.drawContours(contour_image, [c], -1, (0, 255, 0), 3)
    cv2.imwrite(output_path, contour_image)


def plot_contour_comparison(c1, c2, output_path="contour_comp.png"):
    """Plot two contours for comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.plot(c1[:, 0], c1[:, 1], "bo-", label="Contour 1")
    ax.plot(c2[:, 0], c2[:, 1], "ro-", label="Contour 2")

    for i, (x, y) in enumerate(c1):
        ax.text(x, y, str(i), color="black", fontsize=8, ha="right", va="bottom")

    for i, (x, y) in enumerate(c2):
        ax.text(x, y, str(i), color="black", fontsize=8, ha="left", va="top")

    plt.tight_layout()
    plt.savefig(output_path)


def load_standard_contours(path="standard_contours.pkl"):
    with open(path, "rb") as f:
        standard_contours = pickle.load(f)
    return standard_contours


def dfs_to_image(dfs, titles=None, output_path="tables.png"):
    import matplotlib.pyplot as plt

    n = sum([len(df) for df in dfs])
    h = max(0.3 * n, 4)
    fig, axes = plt.subplots(nrows=len(dfs), figsize=(8, h))
    if n == 1:
        axes = [axes]

    for i, (df, ax) in enumerate(zip(dfs, axes)):
        tmpdf = df.reset_index()
        tmpdf = tmpdf.astype(str).replace("<NA>", "")  # or .replace("nan", "") if needed
        tmpdf = tmpdf.astype(str).replace("nan", "")

        ax.axis("off")
        ax.set_title(titles[i] if titles else f"Table {i+1}", fontsize=10)

        table = ax.table(cellText=tmpdf.values, colLabels=tmpdf.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(6)

        num_rows = len(tmpdf)
        num_cols = len(tmpdf.columns)
        for row in range(num_rows + 1):  # +1 for header
            for col in range(num_cols):
                width = 0.2 if col == 0 else 0.04
                table[(row, col)].set_width(width)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def df_to_str(df):
    header = [df.index.name or ""] + list(df.columns)
    body = [[idx] + list(row) for idx, row in zip(df.index, df.values)]

    table_str = t2a(
        header=header,
        body=body,
        style=PresetStyle.thin_compact,
        alignments=[Alignment.LEFT] + [Alignment.RIGHT] * (len(header) - 1),
    )

    return table_str


def string_edit_distance(s1, s2):
    """Calculate the edit distance between two strings.
    https://stackoverflow.com/questions/2460177/edit-distance-in-python
    """
    s1 = "".join(s1.split()).lower()
    s2 = "".join(s2.split()).lower()
    if len(s1) < len(s2):
        return string_edit_distance(s2, s1)

    distances = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        new_distances = [i + 1]
        for j, c2 in enumerate(s2):
            if c1 == c2:
                new_distances.append(distances[j])
            else:
                new_distances.append(min(distances[j], distances[j + 1], new_distances[-1]) + 1)
        distances = new_distances
    return distances[-1] / max(len(s1), len(s2))  # normalize by length
