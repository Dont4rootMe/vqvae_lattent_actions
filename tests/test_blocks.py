import math

import torch

from vqvae_latent_actions.models.blocks import Attention


def test_qk_norm_bounds_the_attention_logits_whatever_the_projections_do():
    """Long production runs grew one decoder attention logit to 3e7 (entropy 0.005) and one of them diverged.
    Layer-normalized q and k have norm sqrt(head_dim), so their scaled dot product cannot exceed sqrt(head_dim)
    while the norm gains are 1."""
    torch.manual_seed(0)
    dim, heads = 64, 4
    x = torch.randn(3, 7, dim)
    for qk_norm in (False, True):
        attn = Attention(dim, heads, qk_norm=qk_norm)
        with torch.no_grad():
            attn.to_q.weight.mul_(1e3)
            attn.to_kv.weight.mul_(1e3)
        biggest = float(attn.logits(x, x).abs().max())
        if qk_norm:
            assert biggest <= math.sqrt(dim // heads) + 1e-4
        else:
            assert biggest > 1e3


def test_logits_are_exactly_what_forward_attends_with():
    """The diagnostic must measure the forward pass, not a lookalike."""
    torch.manual_seed(0)
    dim, heads = 32, 4
    x, context = torch.randn(2, 5, dim), torch.randn(2, 9, dim)
    mask = torch.rand(5, 9) > 0.3
    mask[:, 0] = True
    for qk_norm in (False, True):
        attn = Attention(dim, heads, qk_norm=qk_norm).eval()
        weights = attn.logits(x, context, mask).softmax(-1)
        _, _, v = attn._qkv(x, context)
        manual = attn.proj((weights @ v).transpose(1, 2).reshape(2, 5, dim))
        torch.testing.assert_close(attn(x, context, mask), manual, rtol=1e-4, atol=1e-5)
