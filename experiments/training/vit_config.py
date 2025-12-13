import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

VISION_CONFIG = {
    "batch_size": 1,
    "img_size": 512,          # 512 × 512 images
    "patch_size": 4,          # 4 × 4 patches
    "num_channels": 3,
    "num_patches": 16384,     # (512 / 4)^2 = 128^2 = 16384 tokens
    "num_heads": 8,
    "embed_dim": 512,
    "mlp_dim": 2048,
    "transformer_units": 8,
    "drop_rate": 0.1,
    "num_classes": 50,        # we restrict to 50 classes
    "qkv_bias": False,
    "K": 6,
    "L": 6,
    "M": 1,
}

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]