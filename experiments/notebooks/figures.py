r"""Sizes and styles for the manuscript's figures, after https://jwalton.info/Embed-Publication-Matplotlib-Latex/.

Figures are drawn at their final size and included with `\includegraphics` and no `width=`, so their fonts stay
at the document's sizes.
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt

# The manuscript's geometry in TeX points, from `\showthe\textwidth` and `\showthe\columnsep`.
TEXT_WIDTH_PT = 489.50787
COLUMN_SEP_PT = 24.0
COLUMN_WIDTH_PT = (TEXT_WIDTH_PT - COLUMN_SEP_PT) / 2

PT_PER_INCH = 72.27  # TeX points
GOLDEN = (5**0.5 - 1) / 2


def set_size(
    width_pt: float,
    fraction: float = 1.0,
    subplots: tuple[int, int] = (1, 1),
    height_ratio: float = 1.0,
) -> tuple[float, float]:
    """Figure `(width, height)` in inches: golden-ratio panels, `height_ratio` stretching the height."""
    rows, columns = subplots
    width_in = width_pt * fraction / PT_PER_INCH
    height_in = width_in * GOLDEN * (rows / columns) * height_ratio
    return (width_in, height_in)


# Okabe-Ito, legible under colour-vision deficiency; `clean` is the reference and stays grey.
COLORS = {
    "clean": "#6b6b6b",
    "missing": "#009E73",
    "typo": "#0072B2",
    "outlier": "#CC79A7",
    "swap": "#E69F00",
    "blank_record": "#D55E00",
    "wrong_unit": "#56B4E9",
    "ward_intake": "#000000",
}
ORDER = list(COLORS)
# A series that did not move.
INERT = "#c9c9c9"

# Dashes break the palette's two closest pairs, the blues and the oranges, which thin lines no longer separate.
SCENARIO_DASHES = {
    "clean": (0, ()),
    "missing": (0, ()),
    "typo": (0, ()),
    "outlier": (0, (1, 1.4)),
    "swap": (0, ()),
    "blank_record": (0, (4, 1.2, 1, 1.2)),
    "wrong_unit": (0, (3.4, 1.4)),
    "ward_intake": (0, (5, 1.4)),
}

# The downstream classifier, in a hue no scenario uses.
CLASSIFIER = "#5D3A9B"

LEGEND_SIZE = 7

# The document's own fonts and sizes: 9pt body, 8pt captions; legends and in-panel labels a size below.
PAPER_STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{libertine}",
    "pgf.rcfonts": False,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "figure.labelsize": 9,
    "legend.fontsize": LEGEND_SIZE,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.4,
    "legend.labelspacing": 0.25,
    "legend.borderaxespad": 0.2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
}


def as_tex(name: str) -> str:
    r"""A name in `\texttt`; `\textunderscore`, because matplotlib's usetex path drops the rule `\_` draws."""
    if not plt.rcParams["text.usetex"]:
        return name
    spelled = name.replace("_", r"\textunderscore ")
    return rf"\texttt{{{spelled}}}"


def measure_label(measure: str) -> str:
    """A measure's class name in the standards' sentence case: `DataAccuracyRange` -> `Data accuracy range`."""
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", measure)
    return spaced.capitalize()


def scenario_label(scenario: str) -> str:
    """A scenario key as the manuscript spells it: `blank_record` -> `blank record`."""
    return scenario.replace("_", " ")


def as_percent(rate: float) -> str:
    """A rate as a percentage tick label, escaped when LaTeX sets the text."""
    sign = r"\%" if plt.rcParams["text.usetex"] else "%"
    return f"{round(rate * 100):g}{sign}"
