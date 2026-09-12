import numpy as np
import torch

from vqvae_latent_actions.data.chunks import (batch_to_inputs, build_eval_set, layout_from_manifest, load_eval_set,
                                              save_eval_set, train_loader)


def test_batch_to_inputs_mask_semantics(tiny_manifest, layout):
    loader = train_loader(tiny_manifest, None, batch_size=8, num_workers=0, seed=1, epoch_length=32)
    batch = next(iter(loader))
    actions, mask, emb = batch_to_inputs(batch)
    assert actions.shape == (8, 10, layout.total_dim) and actions.dtype == torch.float32
    assert mask.shape == actions.shape and mask.dtype == torch.bool
    assert emb.shape == (8,) and set(emb.tolist()) <= {0, 1}
    per_step = mask.sum(dim=2)
    assert set(per_step[per_step > 0].tolist()) == {2}          # two real joint dims inside the 119-dim layout
    assert torch.count_nonzero(actions[~mask]) == 0              # padded entries are zeros


def test_train_loader_is_deterministic_for_a_seed(tiny_manifest):
    def first_batch(seed):
        loader = train_loader(tiny_manifest, None, batch_size=4, num_workers=0, seed=seed, epoch_length=16)
        return batch_to_inputs(next(iter(loader)))[0]
    torch.testing.assert_close(first_batch(3), first_batch(3))
    assert not torch.allclose(first_batch(3), first_batch(4))


def test_layout_from_manifest_matches_the_yaml(tiny_manifest, layout):
    assert layout_from_manifest(tiny_manifest).intervals() == layout.intervals()


def test_eval_set_roundtrip_and_cap(tiny_eval_set, tmp_path, layout):
    es = tiny_eval_set
    assert len(es) > 0 and es.actions.shape[1:] == (10, layout.total_dim)
    counts = np.bincount(es.embodiment_index, minlength=2)
    assert counts.tolist() == [6, 6]                             # per_embodiment cap honoured
    assert es.valid.shape == es.actions.shape and es.time_valid.shape == es.actions.shape[:2]
    save_eval_set(es, tmp_path / "e.npz")
    again = load_eval_set(tmp_path / "e.npz")
    np.testing.assert_array_equal(again.actions, es.actions)
    np.testing.assert_array_equal(again.valid, es.valid)
    assert again.embodiment_ids == es.embodiment_ids and again.manifest_sha == es.manifest_sha
    total = sum(len(a) for a, _, _, _ in again.batches(5))
    assert total == len(again)
