import torch

from jula.evaluators.ner_metric import NERMetric
from jula.utils.constants import NE_TAGS

ne_tag_predictions = torch.tensor(
    [
        [
            NE_TAGS.index("O"),
            NE_TAGS.index("O"),
            NE_TAGS.index("B-ORGANIZATION"),
            NE_TAGS.index("I-ORGANIZATION"),
            NE_TAGS.index("B-PERSON"),
            NE_TAGS.index("I-PERSON"),
            NE_TAGS.index("B-LOCATION"),
            NE_TAGS.index("B-ARTIFACT"),
            NE_TAGS.index("I-ARTIFACT"),
            NE_TAGS.index("O"),
        ]
    ],
    dtype=torch.long,
)
ne_tags = torch.tensor(
    [
        [
            NE_TAGS.index("O"),
            NE_TAGS.index("O"),
            NE_TAGS.index("B-ORGANIZATION"),
            NE_TAGS.index("I-ORGANIZATION"),
            NE_TAGS.index("B-PERSON"),
            NE_TAGS.index("O"),
            NE_TAGS.index("O"),
            NE_TAGS.index("B-ARTIFACT"),
            NE_TAGS.index("I-ARTIFACT"),
            NE_TAGS.index("O"),
        ]
    ],
    dtype=torch.long,
)


def test_ner_metric():
    ner_metric = NERMetric()
    ner_metric_args = {
        "example_ids": torch.as_tensor([0]),
        "ne_tag_predictions": ne_tag_predictions,
        "ne_tags": ne_tags,
    }
    ner_metric.update(**ner_metric_args)
    for attr in ner_metric_args.keys():
        setattr(
            ner_metric,
            attr,
            torch.cat(getattr(ner_metric, attr), dim=0),
        )

    results = ner_metric.compute()

    prec, rec = 2 / 4, 2 / 3  # スパンレベル
    f1 = 2 * prec * rec / (prec + rec)
    assert abs(results["ner_f1"] - f1) < 1e-8
