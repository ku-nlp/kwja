import pytest
from omegaconf import DictConfig, ListConfig

from kwja.models.base import filter_dict_items


@pytest.mark.parametrize(
    "item, keys_to_ignore, expected",
    [
        (
            DictConfig({"key_0": "value_0", "key_1": "value_1"}),
            ListConfig(["key_0"]),
            DictConfig({"key_1": "value_1"}),
        ),
        (
            DictConfig({"key": "value"}),
            ListConfig(["key"]),
            DictConfig({}),
        ),
        (
            DictConfig({"key_0": {"key_0_0": "value_0_0", "key_0_1": "value_0_1"}}),
            ListConfig([{"key_0": ["key_0_0"]}]),
            DictConfig({"key_0": {"key_0_1": "value_0_1"}}),
        ),
        (
            DictConfig({"key_0": {"key_0_0": "value_0_0", "key_0_1": "value_0_1"}}),
            ListConfig([{"key_0": ["key_0_0", "key_0_1"]}]),
            DictConfig({"key_0": {}}),
        ),
    ],
)
def test_filter_dict_items(item: DictConfig, keys_to_ignore: ListConfig, expected: DictConfig) -> None:
    assert filter_dict_items(item, keys_to_ignore) == expected
