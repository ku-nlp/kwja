from pathlib import Path

import pytest
import torch
from transformers import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model as TransformersDebertaV2Model

from kwja.modules.components.deberta_v2 import DebertaV2Model


@pytest.mark.parametrize(
    ("position_buckets", "share_att_key", "embedding_size"),
    [(8, True, 24), (-1, False, 12)],
)
def test_matches_transformers(position_buckets: int, share_att_key: bool, embedding_size: int) -> None:
    config = DebertaV2Config(
        vocab_size=64,
        hidden_size=24,
        embedding_size=embedding_size,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=48,
        max_position_embeddings=32,
        position_buckets=position_buckets,
        max_relative_positions=32,
        relative_attention=True,
        pos_att_type=["c2p", "p2c"],
        share_att_key=share_att_key,
        norm_rel_ebd="layer_norm",
        position_biased_input=False,
        conv_kernel_size=3,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    reference = TransformersDebertaV2Model(config).eval()
    model = DebertaV2Model(config).eval()
    torch.testing.assert_close(model.embeddings.position_ids, reference.embeddings.position_ids)
    model.load_state_dict(reference.state_dict(), strict=True)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
        "output_hidden_states": True,
        "output_attentions": True,
    }
    expected = reference(**inputs)
    actual = model(**inputs)
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, rtol=0, atol=0)
    for actual_hidden, expected_hidden in zip(actual.hidden_states, expected.hidden_states, strict=True):
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)
    for actual_attention, expected_attention in zip(actual.attentions, expected.attentions, strict=True):
        torch.testing.assert_close(actual_attention, expected_attention, rtol=0, atol=0)
    actual.last_hidden_state.square().sum().backward()
    expected.last_hidden_state.square().sum().backward()
    for actual_parameter, expected_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


def test_special_token_positions() -> None:
    config = DebertaV2Config(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=16,
        max_position_embeddings=16,
        relative_attention=True,
        position_buckets=8,
        max_relative_positions=16,
        pos_att_type=["c2p", "p2c"],
    )
    model = DebertaV2Model(config)
    actual = model.encoder.get_rel_pos(torch.zeros(2, 4, 8), torch.tensor([[3], [2]]))
    expected = torch.tensor(
        [
            [[0, -1, -2, -8], [1, 0, -1, -8], [2, 1, 0, -8], [-8, -8, -8, 0]],
            [[0, -1, -8, -3], [1, 0, -8, -2], [-8, -8, 0, -8], [3, 2, -8, 0]],
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_load_pretrained(tmp_path: Path) -> None:
    config = DebertaV2Config(
        vocab_size=16, hidden_size=8, num_attention_heads=2, num_hidden_layers=1, intermediate_size=16
    )
    reference = TransformersDebertaV2Model(config)
    reference.save_pretrained(tmp_path)
    model = DebertaV2Model.from_pretrained(tmp_path)
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
