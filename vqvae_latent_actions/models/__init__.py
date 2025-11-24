"""Model registry."""
from .fsq_vqvae import FSQVQVAE, FSQQuantizer, FSQVQVAEOutput
from .lfq_vqvae import LFQVQVAE, LFQQuantizer, LFQVQVAEOutput
from .gmb_vqvae import GMBVQVAE, GMBQuantizer, GMBVQVAEOutput

__all__ = [
    "FSQVQVAE", "FSQQuantizer", "FSQVQVAEOutput",
    "LFQVQVAE", "LFQQuantizer", "LFQVQVAEOutput",
    "GMBVQVAE", "GMBQuantizer", "GMBVQVAEOutput",
]
