"""Model registry."""
from .fsq_vqvae import FSQVQVAE, FSQQuantizer, FSQVQVAEOutput
from .lfq_vqvae import LFQVQVAE, LFQQuantizer, LFQVQVAEOutput

__all__ = ["FSQVQVAE", "FSQQuantizer", "FSQVQVAEOutput", "LFQVQVAE", "LFQQuantizer", "LFQVQVAEOutput"]
