from __future__ import annotations

from typing import Any, Literal

import narwhals as nw

from dqmeasure.base import PositionalMeasure, _require_column

# Scales the median absolute deviation to estimate the standard deviation of a normal distribution
# (1 / Phi^-1(3/4)), so ``threshold`` reads in the familiar sigma units.
_MAD_TO_SIGMA = 1.4826


class RiskOfDataSetInaccuracy(PositionalMeasure):
    """ISO/IEC 25024 `Acc-I-4` "Risk of data set inaccuracy" (`Acc-ML-4` in ISO/IEC 5259-2).

    Column measure, tier 1, positional: unit = cell, subject = the column. The standard
    defines `Acc-I-4` as the risk of inaccuracy, counting outliers, so we report ``1 - X`` to keep every
    measure higher-is-better: ``X`` is the ratio of values that are not outliers.

    The standard leaves the outlier criterion open, and this implementation uses the robust z-score: a value is
    an outlier when ``|value - center| > threshold * scale``, with ``center`` and ``scale`` learned from clean data
    as the median and the sigma-scaled median absolute deviation (see https://en.wikipedia.org/wiki/Robust_measures_of_scale
    and https://en.wikipedia.org/wiki/Median_absolute_deviation).

    When a column is constant in the clean data (``scale = 0``), every deviating value counts as an outlier.

    Parameters
    ----------
    column:
        The numeric column the measure applies to.
    center, scale:
        The reference location and dispersion, or ``None`` to learn them from the clean data at
        [`fit`][dqmeasure.base.BaseMeasure.fit]. Specify both to skip ``fit`` entirely.
    method:
        How the reference is estimated from the clean data. Currently ``"mad"`` (median and sigma-scaled median
        absolute deviation), which is the default.
    threshold:
        How many scale units a value may deviate from the center before it counts as an outlier. The default
        ``3.5`` is the customary cutoff for robust z-scores (Iglewicz-Hoaglin).
    """

    iso_5259_id = "Acc-ML-4"
    iso_25024_id = "Acc-I-4"
    reference_params = ("center", "scale")

    center_: float
    scale_: float

    def __init__(
        self,
        column: str,
        center: float | None = None,
        scale: float | None = None,
        method: Literal["mad"] = "mad",
        threshold: float = 3.5,
    ) -> None:
        super().__init__(column=column)
        self.center = center
        self.scale = scale
        self.method = method
        self.threshold = threshold

    def _validate(self, frame: nw.DataFrame[Any]) -> None:
        _require_column(frame, self.column, numeric=True)

    def _fit_reference(self, frame: nw.DataFrame[Any]) -> dict[str, Any]:
        if self.method != "mad":
            raise ValueError(f"Unsupported method: {self.method!r}")
        col = frame[self.column].drop_nulls().cast(nw.Float64)
        median = float(col.median())
        return {"center": median, "scale": float((col - median).abs().median()) * _MAD_TO_SIGMA}

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        # Per-cell condition: 1.0 if the value is not an outlier, 0.0 if it is, null if the value is missing
        # (the same null-preserving float encoding as DataAccuracyRange, and for the same backend-parity reasons).
        expr = (
            nw.when(~nw.col(self.column).is_null())
            .then(((nw.col(self.column) - self.center_).abs() <= self.threshold * self.scale_).cast(nw.Float64))
            .otherwise(nw.lit(None))
            .alias(self.column)
        )
        return frame.select(expr)[self.column]
