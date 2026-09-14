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



def test_online_logger_continues_the_same_experiment_after_a_restart(tmp_path, monkeypatch):
    """A preempted low-priority job reruns from scratch as a process; without the stored key every attempt would
    open a separate Comet experiment and the curve would be split across them."""
    import sys
    import types

    class Fresh:
        count = 0

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            Fresh.count += 1
            self.key = f"key{Fresh.count}"

        def get_key(self):
            return self.key

        def set_name(self, name): pass
        def add_tags(self, tags): pass
        def log_parameters(self, params): pass
        def log_metrics(self, metrics, step): pass
        def end(self): pass

    class Existing(Fresh):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.key = kwargs["experiment_key"]

    monkeypatch.setitem(sys.modules, "comet_ml", types.SimpleNamespace(Experiment=Fresh, ExistingExperiment=Existing,
                                                                        OfflineExperiment=Fresh))
    key_path = tmp_path / "comet_experiment_key"
    first = RunLogger(mode="online", project="p", resume_key_path=key_path, experiment_name="run")
    assert type(first.experiment) is Fresh and key_path.read_text() == "key1"
    second = RunLogger(mode="online", project="p", resume_key_path=key_path, experiment_name="run")
    assert type(second.experiment) is Existing and second.experiment.get_key() == "key1"
