# Dense Qwen2.5 tensor runner references

The runner and cache/state lifecycle in this directory are KVPilot implementations.
PyTorch is used only for tensor arithmetic; `tokenizers` parses/tokenizes local
files; `safetensors` reads local tensor files. No Transformers model, generation,
or cache implementation is imported or executed.

Architecture details were checked against these primary sources:

- Qwen's [Qwen2.5-0.5B-Instruct configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json).
- The Apache-2.0 Hugging Face/Qwen [Qwen2 mathematical reference at Transformers v4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py): Q/K/V biases, grouped attention, split-half rotary positions, FP32 RMS normalization, and SwiGLU/residual order.
- [PyTorch tensor documentation](https://pytorch.org/docs/stable/torch.html).

The implementation expresses these equations independently and does not vendor
the referenced model classes or cache code. Checkpoint weights and tokenizer
files retain their original licenses. A local SHA256 manifest identifies the
files actually loaded; the runner does not download them or assert that locally
computed hashes authenticate an upstream publisher.

Numerical scope: CPU runs use FP32 weights/state; CUDA and MPS use FP16
weights/state. Attention score, softmax, and weighted-value reductions use FP32.
This is an eager correctness-oriented runner, not a claim of parity with a
production fused serving kernel, nor a native vLLM/Ollama performance baseline.
