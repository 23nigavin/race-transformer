import random
import numpy as np
import torch

# ==================================================
# 0) Config
# ==================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

DATASET_NAME = "ccdv/arxiv-classification"
TEXT_FIELD   = "text"
LABEL_FIELD  = "label"

# ---- Target sequence length and minimum raw length ----
PACK_TARGET_LEN    = 64_000   # final packed sequence length (context size)
MIN_DOC_LEN        = 6_000    # min raw length for individual docs used for packing
PACK_MIN_FRAC      = 0.8      # keep packed seqs >= 0.8 * PACK_TARGET_LEN

# How many docs you'd *like* total (before packing); actual may be smaller
DESIRED_TRAIN_TOTAL = 16000
DESIRED_TEST_TOTAL  = 6000

TEXT_CONFIG = {
    "max_len": PACK_TARGET_LEN,  # model context length (pad/truncate to this)
    "vocab_limit": 50_000,
    "embed_dim": 256,
    "num_heads": 4,    # your choice
    "mlp_dim": 1024,
    "num_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,

    # RACE params
    "K": 5,
    "L": 5,
    "M": 1,

    # Performer params
    "m_features": 256,
    "favor_seed": None,

    # training
    "batch_size": 2,
    "epochs": 100,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "grad_accum_steps": 16,
}
