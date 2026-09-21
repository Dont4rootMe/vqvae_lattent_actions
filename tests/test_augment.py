from __future__ import annotations

import pytest
import torch


def _batch(layout, b=64, t=10, groups=(0, 5, 10), seed=0):
    """Chunks with a few present groups each; everything else is padding."""
    g = torch.Generator().manual_seed(seed)
    membership = layout.membership()
    mask = torch.zeros(b, t, layout.total_dim, dtype=torch.bool)
    for gi in groups:
        mask |= membership[gi].view(1, 1, -1)
    mask[: b // 4] &= ~membership[groups[-1]].view(1, 1, -1)          # some chunks carry fewer groups
    mask[: b // 8] = membership[groups[0]].view(1, 1, -1).expand(b // 8, t, -1)   # and some only one
    actions = torch.randn(b, t, layout.total_dim, generator=g).masked_fill(~mask, 0.0)
    return actions, mask


def test_disabled_augmentation_is_the_identity(layout):
    from vqvae_latent_actions.training.augment import AugmentConfig, augment
    actions, mask = _batch(layout)
    x, m = augment(actions, mask, layout.membership(), AugmentConfig())
    assert torch.equal(x, actions) and torch.equal(m, mask)


def test_config_is_off_by_default():
    from vqvae_latent_actions.training.augment import AugmentConfig
    assert not AugmentConfig().enabled
    assert AugmentConfig(noise_prob=0.1).enabled


def test_padding_stays_zero_and_the_mask_never_grows(layout):
    from vqvae_latent_actions.training.augment import AugmentConfig, augment
    actions, mask = _batch(layout)
    cfg = AugmentConfig(amplitude_prob=1.0, noise_prob=1.0, noise_sigma=0.1, group_drop_prob=1.0)
    x, m = augment(actions, mask, layout.membership(), cfg, generator=torch.Generator().manual_seed(0))
    assert not (m & ~mask).any()
    assert torch.all(x[~m] == 0.0)


def test_amplitude_scales_each_chunk_by_one_factor_inside_the_range(layout):
    from vqvae_latent_actions.training.augment import AugmentConfig, augment
    actions, mask = _batch(layout)
    cfg = AugmentConfig(amplitude_prob=1.0, amplitude_range=(0.5, 1.5))
    x, m = augment(actions, mask, layout.membership(), cfg, generator=torch.Generator().manual_seed(0))
    assert torch.equal(m, mask)
    for i in range(actions.shape[0]):
        a, y = actions[i][mask[i]], x[i][mask[i]]
        factor = (y / a)[a.abs() > 1e-3]
        assert torch.allclose(factor, factor[0].expand_as(factor), atol=1e-4)
        assert 0.5 - 1e-5 <= float(factor[0]) <= 1.5 + 1e-5


def test_group_drop_removes_exactly_one_group_and_never_the_last(layout):
    from vqvae_latent_actions.training.augment import AugmentConfig, augment, present_groups
    actions, mask = _batch(layout)
    membership = layout.membership()
    before = present_groups(mask, membership)
    x, m = augment(actions, mask, membership, AugmentConfig(group_drop_prob=1.0),
                   generator=torch.Generator().manual_seed(0))
    after = present_groups(m, membership)
    lost = (before & ~after).sum(dim=-1)
    assert torch.all(lost[before.sum(-1) >= 2] == 1)
    assert torch.all(lost[before.sum(-1) == 1] == 0)
    assert torch.all(after.sum(-1) >= 1)


def test_noise_touches_only_real_entries_and_is_bounded(layout):
    from vqvae_latent_actions.training.augment import AugmentConfig, augment
    actions, mask = _batch(layout)
    x, m = augment(actions, mask, layout.membership(), AugmentConfig(noise_prob=1.0, noise_sigma=0.05),
                   generator=torch.Generator().manual_seed(0))
    delta = (x - actions)[mask]
    assert float(delta.abs().max()) < 0.05 * 6
    assert float(delta.std()) > 0.0


def test_same_generator_seed_gives_the_same_batch(layout):
    from vqvae_latent_actions.training.augment import AugmentConfig, augment
    actions, mask = _batch(layout)
    cfg = AugmentConfig(amplitude_prob=0.5, noise_prob=0.5, group_drop_prob=0.5)
    a = augment(actions, mask, layout.membership(), cfg, generator=torch.Generator().manual_seed(3))
    b = augment(actions, mask, layout.membership(), cfg, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_probability_controls_how_many_chunks_change(layout):
    from vqvae_latent_actions.training.augment import AugmentConfig, augment
    actions, mask = _batch(layout, b=2000)
    x, _ = augment(actions, mask, layout.membership(), AugmentConfig(amplitude_prob=0.3),
                   generator=torch.Generator().manual_seed(0))
    changed = (x != actions).flatten(1).any(dim=1).float().mean()
    assert float(changed) == pytest.approx(0.3, abs=0.05)
