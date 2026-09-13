from pathlib import Path

import pytest

from kwja.cli import utils
from kwja.cli.config import ModelSize


def test_download_checkpoint_downloads_from_huggingface_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_hf_hub_download(**kwargs: object) -> str:
        captured_kwargs.update(kwargs)
        local_dir = kwargs["local_dir"]
        filename = kwargs["filename"]
        assert isinstance(local_dir, Path)
        assert isinstance(filename, str)
        path = local_dir / filename
        path.write_text("checkpoint", encoding="utf-8")
        return str(path)

    monkeypatch.setattr(utils, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(utils, "_get_model_version", lambda: "v2.4")

    checkpoint_path = utils.download_checkpoint(
        module="typo",
        model_size=ModelSize.TINY,
        checkpoint_dir=tmp_path,
    )

    assert checkpoint_path == tmp_path / "typo_deberta-v2-tiny-wwm.ckpt"
    assert captured_kwargs == {
        "repo_id": "ku-nlp/kwja-checkpoints",
        "filename": "typo_deberta-v2-tiny-wwm.ckpt",
        "revision": "v2.4",
        "local_dir": tmp_path,
    }


def test_download_checkpoint_skips_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_path = tmp_path / "char_deberta-v2-base-wwm.ckpt"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    def fake_hf_hub_download(**_kwargs: object) -> str:
        raise AssertionError("hf_hub_download should not be called")

    monkeypatch.setattr(utils, "hf_hub_download", fake_hf_hub_download)

    assert (
        utils.download_checkpoint(
            module="char",
            model_size=ModelSize.BASE,
            checkpoint_dir=tmp_path,
        )
        == checkpoint_path
    )


def test_hf_hub_download_disables_progress_temporarily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    progress_events: list[str] = []

    def fake_hf_hub_download(**kwargs: object) -> str:
        local_dir = kwargs["local_dir"]
        filename = kwargs["filename"]
        assert isinstance(local_dir, Path)
        assert isinstance(filename, str)
        path = local_dir / filename
        path.write_text("checkpoint", encoding="utf-8")
        return str(path)

    monkeypatch.setattr(utils, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(utils, "are_progress_bars_disabled", lambda: False)
    monkeypatch.setattr(utils, "disable_progress_bars", lambda: progress_events.append("disable"))
    monkeypatch.setattr(utils, "enable_progress_bars", lambda: progress_events.append("enable"))

    utils._hf_hub_download("word_deberta-v2-tiny.ckpt", "v2.4", tmp_path, progress=False)

    assert progress_events == ["disable", "enable"]
