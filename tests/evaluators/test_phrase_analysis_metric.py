import torch

from jula.evaluators.phrase_analysis_metric import PhraseAnalysisMetric
from jula.utils.constants import BASE_PHRASE_FEATURES, IGNORE_INDEX

"""  Example
# S-ID:0-1
* 0
+ 0 <体言>
国立 <基本句-主辞><基本句-区切>
+ 1 <体言>
国会 <基本句-主辞><基本句-区切>
+ 2 <体言>
図書 <基本句-主辞><基本句-区切>
+ 3 <体言>
館 <基本句-主辞>
に <基本句-区切><文節-区切>
* 4
+ 4 <用言:動><モダリティ-意志><モダリティ-勧誘><節-区切><節-主辞>
行こう <基本句-主辞>
。 <基本句-区切><文節-区切>
EOS
"""

word_feature_predictions = torch.tensor(
    [
        [
            [1, 1, 0],  # 国立
            [1, 1, 0],  # 国会
            [1, 1, 0],  # 図書
            [1, 1, 1],  # 館  (<基本句-区切><文節-区切>と誤って予測)
            [0, 0, 0],  # に
            [1, 0, 0],  # 行こう
            [0, 1, 1],  # 。
        ]
    ],
    dtype=torch.long,
)
word_features = torch.tensor(
    [
        [
            [1, 1, 0],  # 国立
            [1, 1, 0],  # 国会
            [1, 1, 0],  # 図書
            [1, 0, 0],  # 館
            [0, 1, 1],  # に
            [1, 0, 0],  # 行こう
            [0, 1, 1],  # 。
        ]
    ],
    dtype=torch.long,
)

base_phrase_feature_predictions = torch.tensor(
    [
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 国立
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 国会
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 図書
            # 館 (<節-主辞>と誤って予測)
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            # に (IGNORE)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # 行こう (<モダリティ-命令>と誤って予測)
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            # 。 (IGNORED)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ],
    dtype=torch.long,
)
base_phrase_features = torch.tensor(
    [
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 国立
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 国会
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 図書
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 館
            [IGNORE_INDEX] * len(BASE_PHRASE_FEATURES),  # に
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],  # 行こう
            [IGNORE_INDEX] * len(BASE_PHRASE_FEATURES),  # 。
        ]
    ],
    dtype=torch.long,
)

phrase_analysis_metric = PhraseAnalysisMetric()
phrase_analysis_metric_args = {
    "example_ids": torch.as_tensor([0]),
    "word_feature_predictions": word_feature_predictions,
    "word_features": word_features,
    "base_phrase_feature_predictions": base_phrase_feature_predictions,
    "base_phrase_features": base_phrase_features,
}
phrase_analysis_metric.update(**phrase_analysis_metric_args)
for attr in phrase_analysis_metric_args.keys():
    setattr(
        phrase_analysis_metric,
        attr,
        torch.cat(getattr(phrase_analysis_metric, attr), dim=0),
    )
results = phrase_analysis_metric.compute()


def test_phrase_analysis_metric():
    # word features
    prec, rec = 5 / 5, 5 / 5
    assert results["基本句-主辞_f1"] == 2 * prec * rec / (prec + rec)
    prec, rec = 3 / 5, 3 / 5  # スパンレベル
    assert results["基本句-区切_f1"] == 2 * prec * rec / (prec + rec)
    prec, rec = 1 / 2, 1 / 2  # スパンレベル
    assert results["文節-区切_f1"] == 2 * prec * rec / (prec + rec)

    # base phrase features
    prec, rec = 1 / 1, 1 / 1
    assert results["用言:動_f1"] == 2 * prec * rec / (prec + rec)
    prec, rec = 4 / 4, 4 / 4
    assert results["体言_f1"] == 2 * prec * rec / (prec + rec)
    prec, rec = 1 / 1, 1 / 1
    assert results["モダリティ-意志_f1"] == 2 * prec * rec / (prec + rec)
    # prec, rec = 0 / 0, 0 / 1
    assert results["モダリティ-勧誘_f1"] == 0.0
    prec, rec = 1 / 2, 1 / 1
    assert results["節-主辞_f1"] == 2 * prec * rec / (prec + rec)
    prec, rec = 1 / 1, 1 / 1
    assert results["節-区切_f1"] == 2 * prec * rec / (prec + rec)
