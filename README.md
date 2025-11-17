# RACE Transformer (Preliminary Prototype)

This repository contains a preliminary experiment integrating RACE sketching into transformer attention layers.

## Running the Benchmark

1. pip install any dependencies
2. run benchmark with python main.py

This builds the RACE C++ extension and benchmarks four models:
1. GPTModel
2. RACEModel
3. LlamaModel
4. LlamaRACEModel