from typing import Any, MutableMapping, MutableSequence, Union


def filter_dict_items(
    item: MutableMapping[str, Any], keys_to_ignore: MutableSequence[Union[str, MutableMapping[str, MutableSequence]]]
) -> MutableMapping[str, Any]:
    """Filter out dictionary items whose key is in keys_to_ignore recursively."""
    for key, value in item.items():
        ignore = False
        for key_to_ignore in keys_to_ignore:
            if isinstance(key_to_ignore, str) and key == key_to_ignore:
                ignore = True
                break
            elif isinstance(key_to_ignore, MutableMapping) and key in key_to_ignore:
                item[key] = filter_dict_items(value, key_to_ignore[key])
                break
        if ignore is True:
            del item[key]
    return item
