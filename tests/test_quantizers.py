import math

import pytest
import torch

from vqvae_latent_actions.models.quantizers import (FSQ, GumbelQuantizer, VQEMA, build_quantizer, code_usage)


def test_fsq_vocabulary_grid_and_roundtrip():
    q = FSQ(levels=[8, 8, 8, 4])
    assert q.vocab_size == 2048 and q.code_dim == 4 and q.out_dim == 4
    assert q.bits_per_token == pytest.approx(11.0)
    z = torch.randn(3, 5, 4) * 3
    out = q(z)
    assert out.codes.shape == z.shape and out.indices.shape == (3, 5)
    assert int(out.indices.min()) >= 0 and int(out.indices.max()) < q.vocab_size
    torch.testing.assert_close(q.indices_to_codes(out.indices), out.codes)          # index <-> code agree
    grid = q.indices_to_codes(torch.arange(q.vocab_size))                            # every code is distinct
    assert grid.shape == (q.vocab_size, 4)
    assert len({tuple(row.tolist()) for row in grid}) == q.vocab_size
    saturated = q(torch.randn(16, 32, 4) * 50)                                       # extreme latents stay in range
    assert int(saturated.indices.min()) >= 0 and int(saturated.indices.max()) < q.vocab_size


def test_fsq_straight_through_gradient_and_commitment():
    q = FSQ(levels=[4, 4], commitment=0.25)
    z = torch.randn(2, 3, 2, requires_grad=True)
    out = q(z)
    (out.codes.sum() + out.aux_loss).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert float(out.aux_loss) >= 0.0


def test_vqema_learns_clusters_and_reports_indices():
    torch.manual_seed(0)
    q = VQEMA(vocab_size=8, code_dim=3, decay=0.8)
    centers = torch.randn(8, 3) * 4
    def batch():
        idx = torch.randint(0, 8, (64,))
        return (centers[idx] + torch.randn(64, 3) * 0.05).view(4, 16, 3)
    q.eval()
    start = batch()
    first = float(((q(start).codes - start) ** 2).mean())     # untrained codebook, before any initialisation
    q.train()
    for _ in range(40):
        q(batch())
    errors = []
    for _ in range(5):
        sample = batch()
        errors.append(((q(sample).codes - sample) ** 2).mean())
    err = float(torch.stack(errors).mean())
    assert err < first and err < 0.05                                  # k-means++ init + EMA land on the clusters
    distance_to_nearest_code = torch.cdist(centers, q.codebook).min(dim=1).values
    assert float(distance_to_nearest_code.max()) < 0.5                 # every cluster is covered by some code
    out = q(batch())
    assert out.indices.shape == (4, 16) and int(out.indices.max()) < 8
    torch.testing.assert_close(q.indices_to_codes(out.indices), q.codebook[out.indices])
    q.eval()
    before = q.codebook.clone()
    q(batch())
    assert torch.equal(before, q.codebook)                             # eval mode never updates the codebook


def test_gumbel_is_hard_and_deterministic_in_eval():
    q = GumbelQuantizer(vocab_size=16, code_dim=8, out_dim=8, temperature=1.0)
    z = torch.randn(2, 4, 8, requires_grad=True)
    q.train()
    out = q(z)
    assert out.codes.shape == (2, 4, 8) and out.indices.shape == (2, 4)
    out.codes.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    q.eval()
    a, b = q(z.detach()), q(z.detach())
    assert torch.equal(a.indices, b.indices)
    torch.testing.assert_close(a.codes, q.indices_to_codes(a.indices))


def test_build_quantizer_and_usage():
    q = build_quantizer({"type": "fsq", "levels": [4, 4, 4]})
    assert isinstance(q, FSQ) and q.vocab_size == 64
    assert isinstance(build_quantizer({"type": "vq", "vocab_size": 32, "code_dim": 4}), VQEMA)
    with pytest.raises(ValueError):
        build_quantizer({"type": "nope"})
    usage = code_usage(torch.tensor([[0, 1], [1, 1]]), vocab_size=4)
    assert usage["codes_used"] == 2 and usage["usage_percent"] == pytest.approx(50.0)
    assert usage["perplexity"] == pytest.approx(math.exp(-(0.25 * math.log(0.25) + 0.75 * math.log(0.75))), rel=1e-5)


def test_lfq_optional():
    pytest.importorskip("vector_quantize_pytorch")
    q = build_quantizer({"type": "lfq", "vocab_size": 256})
    z = torch.randn(2, 3, q.code_dim)
    out = q(z)
    assert q.vocab_size == 256 and q.code_dim == 8
    assert out.indices.shape == (2, 3) and int(out.indices.max()) < 256


def test_fsq_bound_is_the_code_without_rounding():
    """`bound` is what the decoder must see while the quantizer is warming up: the same value range it will get
    once the grid is switched on. Without it the encoder drifts into tanh saturation and most levels go unused."""
    q = FSQ(levels=[8, 8, 8, 4])
    z = torch.randn(4, 6, 4) * 3
    bounded = q.bound(z)
    assert bounded.shape == z.shape
    torch.testing.assert_close(torch.round(bounded * q._half_width) / q._half_width, q(z).codes)
    grid = q.indices_to_codes(torch.arange(q.vocab_size))
    lo, hi = float(grid.min()), float(grid.max())
    for scale in (1.0, 50.0):
        b = q.bound(torch.randn(8, 8, 4) * scale)
        assert lo - 0.01 <= float(b.min()) and float(b.max()) <= hi + 0.01
        assert torch.isfinite(b).all()


def test_bound_defaults_to_identity_for_learned_codebooks():
    """A learned codebook follows whatever scale the encoder settles on, so it needs no bounding."""
    q = VQEMA(vocab_size=8, code_dim=3)
    z = torch.randn(2, 5, 3)
    torch.testing.assert_close(q.bound(z), z)


def test_fsq_saturation_reports_how_much_of_the_grid_is_out_of_reach():
    q = FSQ(levels=[8, 8, 8, 4])
    assert q.saturation(torch.zeros(4, 4)) == 0.0
    assert q.saturation(torch.full((4, 4), 50.0)) == 1.0
