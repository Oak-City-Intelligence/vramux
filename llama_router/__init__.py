"""llama.cpp backend for vramux.

Provides an ollama-compatible HTTP surface (registered on :11434) backed by
a supervised llama-server process. Lets every existing ollama caller keep
working unchanged while the actual inference is served by llama.cpp.
"""
