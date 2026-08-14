from __future__ import annotations

from typing import Any

import narwhals as nw

from dqmeasure.base import PositionalMeasure


class LabelCompleteness(PositionalMeasure):
    """ISO/IEC 5259-2 `Com-ML-5` "Label completeness".

    Column measure, tier 1, positional: unit = sample (row), subject = the label column.

    Counts unlabelled or incompletely labelled samples ``A`` over all samples ``B`` and reports
    ``X = 1 - A/B``, the fraction of fully labelled samples. A sample counts as unlabelled when its label is
    null: missing labels are assumed to be null-encoded (see the model doc's simplifying
    assumptions). [`predict`][dqmeasure.base.PositionalMeasure.predict] reports ``1.0`` for a labelled sample,
    and its mean is exactly the standard's ``1 - A/B``. There is no
    reference to learn, the measure works without [`fit`][dqmeasure.base.BaseMeasure.fit].

    `Com-ML-5` coincides numerically with `Com-ML-3` feature completeness on the label column; the measures
    stay distinct in the role of the column and their measurement function.

    Parameters
    ----------
    column:
        The label column the measure applies to; any dtype works.
    """

    iso_5259_id = "Com-ML-5"
    iso_25024_id = None

    def _measure_units(self, frame: nw.DataFrame[Any]) -> nw.Series[Any]:
        # Per-sample condition: 1.0 if labelled, 0.0 if the label is null. Every sample is in scope, so the
        # output carries no nulls and score() divides by the row count.
        expr = (~nw.col(self.column).is_null()).cast(nw.Float64).alias(self.column)
        return frame.select(expr)[self.column]
