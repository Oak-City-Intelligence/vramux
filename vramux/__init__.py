"""vramux — one ollama-compatible endpoint in front of several inference
runtimes, arbitrating a single GPU between them.

Clients keep talking ollama's API on :11434. Behind it, a model may be served
by a llama-server subprocess on a local GGUF or by a container that ships its
own OpenAI-compatible server; both compete for the same GPU under the same
swap and idle-unload rules.
"""
