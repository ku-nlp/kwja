import textwrap

import torch
from rhoknp import Document

from jula.evaluators.dependency_parsing_metric import DependencyParsingMetric

document = Document.from_knp(
    textwrap.dedent(
        """\
        # S-ID:000-1 KNP:5.0-2ad4f6df
        * 1D <BGH:風/かぜ><文頭><ガ><助詞><体言><一文字漢字><係:ガ格><区切:0-0><格要素><連用要素><正規化代表表記:風/かぜ><主辞代表表記:風/かぜ>
        + 1D <BGH:風/かぜ><文頭><ガ><助詞><体言><一文字漢字><係:ガ格><区切:0-0><格要素><連用要素><名詞項候補><先行詞候補><正規化代表表記:風/かぜ><主辞代表表記:風/かぜ><解析格:ガ>
        風 かぜ 風 名詞 6 普通名詞 1 * 0 * 0 "代表表記:風/かぜ カテゴリ:抽象物 漢字読み:訓" <代表表記:風/かぜ><カテゴリ:抽象物><漢字読み:訓><正規化代表表記:風/かぜ><漢字><かな漢字><名詞相当語><文頭><自立><内容語><タグ単位始><文節始><文節主辞>
        が が が 助詞 9 格助詞 1 * 0 * 0 NIL <かな漢字><ひらがな><付属>
        * -1D <BGH:吹く/ふく><文末><補文ト><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><動態述語><正規化代表表記:吹く/ふく><主辞代表表記:吹く/ふく>
        + -1D <BGH:吹く/ふく><文末><補文ト><句点><用言:動><レベル:C><区切:5-5><ID:（文末）><係:文末><提題受:30><主節><格要素><連用要素><動態述語><正規化代表表記:吹く/ふく><主辞代表表記:吹く/ふく><用言代表表記:吹く/ふく><節-区切><節-主辞><時制:非過去><主題格:一人称優位><格関係0:ガ:風><格解析結果:吹く/ふく:動1:ガ/C/風/0/0/000-1;ニ/U/-/-/-/-;ト/U/-/-/-/-;デ/U/-/-/-/-;カラ/U/-/-/-/-;時間/U/-/-/-/-><標準用言代表表記:吹く/ふく>
        吹く ふく 吹く 動詞 2 * 0 子音動詞カ行 2 基本形 2 "代表表記:吹く/ふく 補文ト" <代表表記:吹く/ふく><補文ト><正規化代表表記:吹く/ふく><かな漢字><活用語><表現文末><自立><内容語><タグ単位始><文節始><文節主辞><用言表記先頭><用言表記末尾><用言意味表記末尾>
        。 。 。 特殊 1 句点 1 * 0 * 0 NIL <英記号><記号><文末><付属>
        EOS
        """
    )
)
documents = [document]
dependency_type_predictions = torch.tensor(
    [
        [
            [1, 1],  # P
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ]
    ],
    dtype=torch.long,
)


def test_dependency_parsing_metric():
    dependency_parsing_metric = DependencyParsingMetric()
    dependency_parsing_metric_args = {
        "example_ids": torch.as_tensor([0]),
        "dependency_predictions": torch.tensor(
            [
                [
                    [2, 1],  # 風
                    [0, 2],  # が
                    [6, 3],  # 吹く
                    [2, 6],  # 。
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ],
            dtype=torch.long,
        ),
        "dependency_type_predictions": dependency_type_predictions,
    }
    dependency_parsing_metric.update(**dependency_parsing_metric_args)
    for attr in dependency_parsing_metric_args.keys():
        setattr(
            dependency_parsing_metric,
            attr,
            torch.cat(getattr(dependency_parsing_metric, attr), dim=0),
        )

    results = dependency_parsing_metric.compute(documents)

    prec, rec = 2 / 2, 2 / 2
    assert results["base_phrase_UAS_f1"] == 2 * prec * rec / (prec + rec)
    prec, rec = 4 / 4, 4 / 4
    assert results["morpheme_UAS_f1"] == 2 * prec * rec / (prec + rec)
    prec, rec = 1 / 2, 1 / 2
    assert results["base_phrase_LAS_f1"] == 2 * prec * rec / (prec + rec)
    prec, rec = 3 / 4, 3 / 4
    assert results["morpheme_LAS_f1"] == 2 * prec * rec / (prec + rec)


def test_cyclic_dependency():
    dependency_parsing_metric = DependencyParsingMetric()
    dependency_parsing_metric_args = {
        "example_ids": torch.as_tensor([0]),
        "dependency_predictions": torch.tensor(
            [
                [
                    [1, 2],  # 風
                    [2, 0],  # が
                    [0, 1],  # 吹く
                    [2, 6],  # 。
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ],
            dtype=torch.long,
        ),
        "dependency_type_predictions": dependency_type_predictions,
    }
    dependency_parsing_metric.update(**dependency_parsing_metric_args)
    for attr in dependency_parsing_metric_args.keys():
        setattr(
            dependency_parsing_metric,
            attr,
            torch.cat(getattr(dependency_parsing_metric, attr), dim=0),
        )
    """
    cyclic dependency を解消すると
    　　　　　  s   g
    　風が　　 1P  1D
    　吹く。　-1D -1D

    　　　　  s   g
    　風　　 1P  2D
    　が　　 2D  0D
    　吹く　 3D -1D
    　。　　-1D  2D
    となる
    """
    results = dependency_parsing_metric.compute(documents)

    prec, rec = 2 / 2, 2 / 2
    assert results["base_phrase_UAS_f1"] == 2 * prec * rec / (prec + rec)
    # prec, rec = 0 / 4, 0 / 4
    assert results["morpheme_UAS_f1"] == 0.0
    prec, rec = 1 / 2, 1 / 2
    assert results["base_phrase_LAS_f1"] == 2 * prec * rec / (prec + rec)
    # prec, rec = 0 / 4, 0 / 4
    assert results["morpheme_LAS_f1"] == 0.0


def test_multiple_roots():
    dependency_parsing_metric = DependencyParsingMetric()
    dependency_parsing_metric_args = {
        "example_ids": torch.as_tensor([0]),
        "dependency_predictions": torch.tensor(
            [
                [
                    [6, 2],  # 風
                    [6, 0],  # が
                    [6, 3],  # 吹く
                    [6, 2],  # 。
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ],
            dtype=torch.long,
        ),
        "dependency_type_predictions": dependency_type_predictions,
    }
    dependency_parsing_metric.update(**dependency_parsing_metric_args)
    for attr in dependency_parsing_metric_args.keys():
        setattr(
            dependency_parsing_metric,
            attr,
            torch.cat(getattr(dependency_parsing_metric, attr), dim=0),
        )
    """
    multiple roots を解消すると
    　　　　　  s   g
    　風が　　-1P  1D
    　吹く。　 0D -1D

    　　　　  s   g
    　風　　-1P  2D
    　が　　 0D  0D
    　吹く　 3D -1D
    　。　　 1D  2D
    となる
    """
    results = dependency_parsing_metric.compute(documents)

    # prec, rec = 0 / 2, 0 / 2
    assert results["base_phrase_UAS_f1"] == 0.0
    prec, rec = 1 / 4, 1 / 4
    assert results["morpheme_UAS_f1"] == 2 * prec * rec / (prec + rec)
    # prec, rec = 0 / 2, 0 / 2
    assert results["base_phrase_LAS_f1"] == 0.0
    prec, rec = 1 / 4, 1 / 4
    assert results["morpheme_LAS_f1"] == 2 * prec * rec / (prec + rec)


def test_no_root():
    dependency_parsing_metric = DependencyParsingMetric()
    dependency_parsing_metric_args = {
        "example_ids": torch.as_tensor([0]),
        "dependency_predictions": torch.tensor(
            [
                [
                    [2, 1],
                    [0, 2],
                    [0, 3],
                    [2, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ],
            dtype=torch.long,
        ),
        "dependency_type_predictions": dependency_type_predictions,
    }
    dependency_parsing_metric.update(**dependency_parsing_metric_args)
    for attr in dependency_parsing_metric_args.keys():
        setattr(
            dependency_parsing_metric,
            attr,
            torch.cat(getattr(dependency_parsing_metric, attr), dim=0),
        )
    """
    no root を解消すると
    　　　　　  s   g
    　風が　　 1P  1D
    　吹く。　-1D -1D

    　　　　 s  g
    　風　　2P 2D
    　が　　0D 0D
    　吹く　3D 6D
    　。　　6D 2D
    となる
    """
    results = dependency_parsing_metric.compute(documents)

    prec, rec = 2 / 2, 2 / 2
    assert results["base_phrase_UAS_f1"] == 2 * prec * rec / (prec + rec)
    prec, rec = 2 / 4, 2 / 4
    assert results["morpheme_UAS_f1"] == 2 * prec * rec / (prec + rec)
    prec, rec = 1 / 2, 1 / 2
    assert results["base_phrase_LAS_f1"] == 2 * prec * rec / (prec + rec)
    prec, rec = 1 / 4, 1 / 4
    assert results["morpheme_LAS_f1"] == 2 * prec * rec / (prec + rec)
