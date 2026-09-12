import torch

from vqvae_latent_actions.models.hier_tokenizer import HierActionTokenizer
from vqvae_latent_actions.training.evaluate import (evaluate_tokenizer, model_report_extra,
                                                    padding_invariance_mismatch)


def test_evaluate_tokenizer_reports_errors_and_usage(tiny_model_config, tiny_eval_set, device):
    torch.manual_seed(0)
    model = HierActionTokenizer(tiny_model_config).eval()
    summary = evaluate_tokenizer(model, tiny_eval_set, batch_size=4, device=device)
    total = summary["total"]
    assert total["n"] == len(tiny_eval_set)
    assert total["tokens_mean"] == model.num_tokens and total["decode_failures"] == 0
    assert total["rmse"] > 0 and set(summary["per_embodiment"]) == {"toy_joint_0", "toy_joint_1"}
    usage = summary["usage"]
    assert usage["vocab_size"] == model.vocab_size and usage["tokens_per_chunk"] == model.num_tokens
    assert usage["bits_per_chunk"] == model.bits_per_chunk
    assert 0 < usage["usage_percent"] <= 100 and usage["perplexity"] >= 1


def test_padding_invariance_is_exact(tiny_model_config, tiny_eval_set, device):
    torch.manual_seed(0)
    model = HierActionTokenizer(tiny_model_config).eval()
    assert padding_invariance_mismatch(model, tiny_eval_set, device=device) == 0.0


def test_model_report_extra(tiny_model_config):
    model = HierActionTokenizer(tiny_model_config)
    extra = model_report_extra(model)
    assert extra["num_tokens"] == 4 and extra["vocab_size"] == 16 and extra["bits_per_chunk"] == 16.0
    assert extra["quantizer"]["type"] == "fsq" and extra["parameters"] > 0


def test_evaluate_can_skip_quantization_during_warmup(tiny_model_config, tiny_eval_set, device):
    """During the quantizer warmup the model is a plain autoencoder, so scoring it through the grid is
    meaningless; `quantize=False` must score the continuous path and say so."""
    torch.manual_seed(0)
    model = HierActionTokenizer(tiny_model_config).eval()
    quantized = evaluate_tokenizer(model, tiny_eval_set, batch_size=4, device=device)
    continuous = evaluate_tokenizer(model, tiny_eval_set, batch_size=4, device=device, quantize=False)
    assert quantized["usage"]["quantized"] is True
    assert continuous["usage"]["quantized"] is False
    assert continuous["usage"]["codes_used"] == 0 and continuous["usage"]["perplexity"] == 1.0
    assert continuous["total"]["rmse"] < quantized["total"]["rmse"]      # no grid error on the continuous path
    assert continuous["total"]["n"] == len(tiny_eval_set)
