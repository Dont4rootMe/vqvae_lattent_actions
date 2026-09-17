import math

import pytest
import torch

from vqvae_latent_actions.models.hier_tokenizer import HierActionTokenizer, HierTokenizerConfig


def _mask(layout, batch: int, horizon: int, groups: list[str], time_cut: int | None = None) -> torch.Tensor:
    m = torch.zeros(batch, horizon, layout.total_dim, dtype=torch.bool)
    for name in groups:
        s, e = layout.intervals()[name]
        m[:, :, s:e] = True
    if time_cut is not None:
        m[:, time_cut:, :] = False
    return m


@pytest.fixture
def model(tiny_model_config):
    torch.manual_seed(0)
    return HierActionTokenizer(tiny_model_config).eval()


def test_shapes_tokens_and_budget(model, layout):
    b, t, d = 3, 10, layout.total_dim
    mask = _mask(layout, b, t, ["left_arm.joints", "left_arm.gripper"])
    actions = torch.randn(b, t, d) * mask
    tokens, recon = model.encode_decode(actions, mask)
    assert tokens.shape == (b, 4) and tokens.dtype == torch.long
    assert int(tokens.min()) >= 0 and int(tokens.max()) < 16
    assert recon.shape == (b, t, d)
    assert model.num_tokens == 4 and model.vocab_size == 16
    assert model.bits_per_chunk == pytest.approx(4 * math.log2(16))


def test_shorter_horizon_is_accepted(model, layout):
    mask = _mask(layout, 2, 5, ["right_arm.joints"])
    tokens, recon = model.encode_decode(torch.randn(2, 5, layout.total_dim) * mask, mask)
    assert tokens.shape == (2, 4) and recon.shape == (2, 5, layout.total_dim)


def test_padding_values_do_not_influence_the_model(model, layout):
    b, t, d = 2, 10, layout.total_dim
    mask = _mask(layout, b, t, ["left_arm.joints", "left_hand.joints"], time_cut=7)
    actions = torch.randn(b, t, d) * mask
    polluted = actions.clone()
    polluted[~mask] = 1e6
    polluted[0, 9, 0] = float("nan")                       # padded position, must never reach the graph
    tokens_a, recon_a = model.encode_decode(actions, mask)
    tokens_b, recon_b = model.encode_decode(polluted, mask)
    assert torch.equal(tokens_a, tokens_b)
    torch.testing.assert_close(recon_a, recon_b)


def test_decoder_outputs_exact_zeros_in_padding(model, layout):
    mask = _mask(layout, 2, 10, ["torso.joints", "head.joints"], time_cut=6)
    _, recon = model.encode_decode(torch.randn(2, 10, layout.total_dim) * mask, mask)
    assert torch.count_nonzero(recon[~mask]) == 0
    assert recon[mask].abs().sum() > 0


def test_group_queries_only_see_their_own_group(model, layout):
    b, t = 2, 10
    mask = _mask(layout, b, t, ["left_hand.joints", "right_arm.joints"])
    actions = torch.randn(b, t, layout.total_dim) * mask
    changed = actions.clone()
    s, e = layout.intervals()["left_hand.joints"]
    changed[:, :, s:e] += 5.0
    g_hand = layout.names.index("left_hand.joints")
    g_arm = layout.names.index("right_arm.joints")
    step = model.encode_step_tokens(actions, mask, mix=False)             # [B, T, K, d], groups still isolated
    step2 = model.encode_step_tokens(changed, mask, mix=False)
    assert not torch.allclose(step[:, :, g_hand], step2[:, :, g_hand])    # its own group reacts
    torch.testing.assert_close(step[:, :, g_arm], step2[:, :, g_arm])     # a different group does not
    assert not torch.allclose(step[:, :, layout.num_groups:], step2[:, :, layout.num_groups:])  # free queries do
    mixed = model.encode_step_tokens(actions, mask)                       # after the within-step self-attention
    mixed2 = model.encode_step_tokens(changed, mask)
    assert not torch.allclose(mixed[:, :, g_arm], mixed2[:, :, g_arm])    # information crosses groups there


def test_loss_backward_reaches_every_parameter(tiny_model_config, layout):
    torch.manual_seed(0)
    model = HierActionTokenizer(tiny_model_config).train()
    mask = _mask(layout, 4, 10, ["left_arm.joints", "left_arm.gripper", "base.velocity"], time_cut=8)
    actions = torch.randn(4, 10, layout.total_dim) * mask
    out = model(actions, mask)
    assert set(out) >= {"loss", "recon_mse", "aux_loss", "indices", "recon"}
    assert torch.isfinite(out["loss"]) and float(out["recon_mse"]) > 0
    out["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert missing == []


def test_forward_can_skip_quantisation(tiny_model_config, layout):
    """The warmup phase trains the plain autoencoder, so the reconstruction comes from unrounded latents that
    are still held inside the range the grid will impose."""
    torch.manual_seed(0)
    model = HierActionTokenizer(tiny_model_config).eval()
    mask = _mask(layout, 2, 10, ["left_arm.joints", "head.joints"])
    actions = torch.randn(2, 10, layout.total_dim) * mask
    quantised = model(actions, mask)
    plain = model(actions, mask, quantize=False)
    assert float(plain["aux_loss"]) == 0.0
    assert not torch.allclose(quantised["recon"], plain["recon"])
    torch.testing.assert_close(plain["recon"],
                               model.decode_latents(model.quantizer.bound(plain["latents"]), mask))
    assert torch.count_nonzero(plain["recon"][~mask]) == 0


def test_group_weights_change_the_loss(tiny_model_config, layout):
    cfg = HierTokenizerConfig.from_dict({**tiny_model_config.to_dict(), "group_weights": {"left_arm.gripper": 10.0}})
    torch.manual_seed(0)
    weighted = HierActionTokenizer(cfg).eval()
    torch.manual_seed(0)
    plain = HierActionTokenizer(tiny_model_config).eval()
    mask = _mask(layout, 2, 10, ["left_arm.joints", "left_arm.gripper"])
    actions = torch.randn(2, 10, layout.total_dim) * mask
    assert float(weighted(actions, mask)["recon_mse"]) != float(plain(actions, mask)["recon_mse"])


def test_detokenize_matches_forward_and_survives_save_load(model, layout, tmp_path):
    mask = _mask(layout, 2, 10, ["right_hand.joints", "right_arm.gripper"])
    actions = torch.randn(2, 10, layout.total_dim) * mask
    tokens, recon = model.encode_decode(actions, mask)
    torch.testing.assert_close(model.detokenize(tokens, mask), recon)
    model.save_pretrained(tmp_path / "ckpt")
    again = HierActionTokenizer.from_pretrained(tmp_path / "ckpt").eval()
    tokens2, recon2 = again.encode_decode(actions, mask)
    assert torch.equal(tokens, tokens2)
    torch.testing.assert_close(recon, recon2)


def test_warmup_decodes_the_bounded_latents(model, layout):
    """While the quantizer is off the decoder must still see the value range it gets once the grid is on.
    Raw latents let the encoder drift into tanh saturation, which pins most levels and wastes the vocabulary."""
    b, t = 2, 10
    mask = _mask(layout, b, t, ["left_arm.joints", "base.velocity"])
    actions = torch.randn(b, t, layout.total_dim) * mask
    out = model(actions, mask, quantize=False)
    latents = model.encode_continuous(actions, mask)
    bounded = model.quantizer.bound(latents).detach()
    torch.testing.assert_close(out["recon"], model.decode_latents(bounded, mask))
    assert not torch.allclose(out["recon"], model.decode_latents(latents, mask))
    grid = model.quantizer.indices_to_codes(torch.arange(model.vocab_size))
    assert float(bounded.min()) >= float(grid.min()) - 0.01 and float(bounded.max()) <= float(grid.max()) + 0.01



def test_qk_norm_reaches_every_attention_and_old_exports_still_load(tiny_model_config, tmp_path):
    from vqvae_latent_actions.models.blocks import Attention
    new = HierActionTokenizer(HierTokenizerConfig.from_dict({**tiny_model_config.to_dict(), "qk_norm": True}))
    attentions = [m for m in new.modules() if isinstance(m, Attention)]
    assert attentions and all(a.qk_norm for a in attentions)

    legacy = {k: v for k, v in tiny_model_config.to_dict().items() if k != "qk_norm"}   # exported before the option
    old = HierActionTokenizer(HierTokenizerConfig.from_dict(legacy))
    assert not any(a.qk_norm for a in old.modules() if isinstance(a, Attention))
    old.save_pretrained(tmp_path / "old")
    HierActionTokenizer.from_pretrained(tmp_path / "old")
    new.save_pretrained(tmp_path / "new")
    HierActionTokenizer.from_pretrained(tmp_path / "new")
    assert not list(tmp_path.rglob("*.tmp"))              # every file of an export lands with one rename
