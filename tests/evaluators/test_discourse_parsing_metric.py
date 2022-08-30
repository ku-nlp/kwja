import pytest
import torch

from jula.evaluators.discourse_parsing_metric import DiscourseParsingMetric
from jula.utils.constants import DISCOURSE_RELATIONS, IGNORE_INDEX

discourse_parsing_predictions = torch.tensor(
    [
        [
            [
                DISCOURSE_RELATIONS.index("談話関係なし"),
                DISCOURSE_RELATIONS.index("談話関係なし"),
                DISCOURSE_RELATIONS.index("原因・理由"),
                DISCOURSE_RELATIONS.index("根拠"),
            ],
            [
                DISCOURSE_RELATIONS.index("談話関係なし"),
                DISCOURSE_RELATIONS.index("談話関係なし"),
                DISCOURSE_RELATIONS.index("談話関係なし"),
                DISCOURSE_RELATIONS.index("談話関係なし"),
            ],
            [
                DISCOURSE_RELATIONS.index("談話関係なし"),
                DISCOURSE_RELATIONS.index("談話関係なし"),
                DISCOURSE_RELATIONS.index("談話関係なし"),
                DISCOURSE_RELATIONS.index("原因・理由"),
            ],
            [
                DISCOURSE_RELATIONS.index("談話関係なし"),
                DISCOURSE_RELATIONS.index("談話関係なし"),
                DISCOURSE_RELATIONS.index("談話関係なし"),
                DISCOURSE_RELATIONS.index("談話関係なし"),
            ],
        ]
    ],
    dtype=torch.long,
)
discourse_parsing_labels = torch.tensor(
    [
        [
            [
                DISCOURSE_RELATIONS.index("談話関係なし"),
                IGNORE_INDEX,
                DISCOURSE_RELATIONS.index("原因・理由"),
                DISCOURSE_RELATIONS.index("根拠"),
            ],
            [
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
                IGNORE_INDEX,
            ],
            [
                DISCOURSE_RELATIONS.index("談話関係なし"),
                IGNORE_INDEX,
                DISCOURSE_RELATIONS.index("談話関係なし"),
                DISCOURSE_RELATIONS.index("談話関係なし"),
            ],
            [
                DISCOURSE_RELATIONS.index("談話関係なし"),
                IGNORE_INDEX,
                DISCOURSE_RELATIONS.index("談話関係なし"),
                DISCOURSE_RELATIONS.index("談話関係なし"),
            ],
        ]
    ],
    dtype=torch.long,
)


def test_discourse_parsing_metric():
    discourse_parsing_metric = DiscourseParsingMetric()
    discourse_parsing_metric_args = {
        "discourse_parsing_predictions": discourse_parsing_predictions,
        "discourse_parsing_labels": discourse_parsing_labels,
    }
    discourse_parsing_metric.update(**discourse_parsing_metric_args)
    for attr in discourse_parsing_metric_args.keys():
        setattr(
            discourse_parsing_metric,
            attr,
            torch.cat(getattr(discourse_parsing_metric, attr), dim=0),
        )

    results = discourse_parsing_metric.compute()

    acc = 8 / 9
    assert results["discourse_parsing_acc"] == pytest.approx(acc)

    prec = 2 / 3
    rec = 2 / 2
    f1 = 2 * prec * rec / (prec + rec)
    assert results["discourse_parsing_precision"] == pytest.approx(prec)
    assert results["discourse_parsing_recall"] == pytest.approx(rec)
    assert results["discourse_parsing_f1"] == pytest.approx(f1)
