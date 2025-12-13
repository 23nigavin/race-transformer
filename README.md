# RACE Transformer (Preliminary Prototype)

This repository contains a preliminary experiment integrating RACE sketching into transformer attention layers.

## Running the Benchmark

1. pip install any dependencies
2. run benchmark with python -m experiments.inference.ttft_prefill

This builds the RACE C++ extension and benchmarks four models:
1. GPTModel
2. RACEModel
3. LlamaModel
4. LlamaRACEModel

## Running the Training Experiments (GPU Recommended)

To run the long context classification experiments on ArXiv:
python -m experiments.training.arxiv_train

To run the ViT and Food 101 experiments:
python -m experiments.training.vit_train
