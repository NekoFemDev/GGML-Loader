#pragma once

#include <string>

// Resolves a model reference to a real GGUF file path, so `run`/`chat`/`bench`/`inspect` can
// take either an on-disk .gguf path (unchanged, most common case) or an Ollama model reference
// (e.g. "qwen2.5-coder:7b-instruct-q4_K_M") pulled locally via `ollama pull` -- Ollama stores the
// raw GGUF weight blob content-addressed under ~/.ollama/models/blobs/ (no .gguf extension) with
// a separate manifest mapping the name:tag to that blob's sha256 digest; this looks the manifest
// up and returns the underlying blob's path directly, since it's byte-identical GGUF our loader
// already reads. If `spec` is an existing file, it's returned unchanged. If it looks like an
// Ollama reference but can't be resolved (not pulled, no ~/.ollama, malformed manifest, etc.),
// `spec` is still returned unchanged (a diagnostic is printed to stderr) so the caller's normal
// "failed to load" error path still fires with a useful message.
std::string resolve_model_ref(const std::string & spec);
