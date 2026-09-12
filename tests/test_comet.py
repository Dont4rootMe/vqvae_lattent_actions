import json

from vqvae_latent_actions.training.comet import RunLogger


def test_disabled_logger_still_writes_jsonl(tmp_path):
    logger = RunLogger(mode="disabled", jsonl_path=tmp_path / "log.jsonl", config={"a": 1})
    assert logger.mode == "disabled"
    logger.log_params({"lr": 0.1})
    logger.log_metrics({"loss": 1.5}, step=3)
    logger.log_metrics({"loss": 1.0}, step=4)
    logger.end()
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [3, 4] and rows[-1]["loss"] == 1.0


def test_offline_mode_creates_an_archive(tmp_path):
    import pytest
    pytest.importorskip("comet_ml")
    logger = RunLogger(mode="offline", project="unit-test", offline_directory=tmp_path / "offline",
                       jsonl_path=tmp_path / "log.jsonl", experiment_name="unit")
    assert logger.mode == "offline"
    logger.log_params({"n": 4})
    logger.log_metrics({"loss": 0.5}, step=1)
    logger.end()
    assert list((tmp_path / "offline").glob("*.zip"))
