# GGML-Loader

A model loader/inference engine aimed at mobile ARM devices with **Mali GPUs**, built on top of
[ggml](https://github.com/ggml-org/ggml). The goal is text (GGUF/LLM) *and* SDXL (image
diffusion) support with a Mali-tuned GPU backend -- llama.cpp alone only covers the text side,
and (see "Findings" below) its generic Vulkan path leaves real performance on the table on Mali
specifically.

**Status: early, but the CPU/Vulkan text-inference path is real now, not just a stub. The
Mali-specific GPU backend -- this project's actual reason to exist -- still hasn't been started.**

## What's actually here right now

- A from-scratch forward pass for four gguf architectures -- **phi3, qwen2, qwen3, llama** --
  written directly against ggml's tensor ops, not a wrapper around llama.cpp. Per-architecture
  differences (fused vs separate QKV, QKV bias, per-head QK-Norm, fused vs separate FFN gate/up,
  tied vs untied output, RoPE variant and scaling) are handled through a small set of flags read
  from each model's own gguf metadata, not a separate code path per model.
- A persistent F16 KV cache, sized per-model from its actual head/layer shape. `run`, `chat`, and
  `bench` auto-size the context window at load time from the RAM still free once the weights are
  resident (minus a 384 MiB reserve and the hybrid SSM state, 20% headroom, rounded down and
  clamped to the model's trained context length); `--ctx N` overrides it.
- A real GPT-2-style byte-level BPE tokenizer: encoder (pretokenize + merge-rank BPE from the
  model's own `tokenizer.ggml.merges`) and decoder, not just a decoder.
- **Both CPU and Vulkan execution**, selectable per run, sharing the identical graph-building
  code -- weights load straight into whichever backend's memory (`ggml_backend_tensor_set`, not
  the CPU-only bulk gguf loading path), so there's no separate "GPU version" to keep in sync.
- A single `ggml-loader` CLI with four subcommands:
  - `ggml-loader inspect <model.gguf>` -- dump GGUF metadata and tensor list.
  - `ggml-loader run <model.gguf> -p "prompt" [-n N] [--backend cpu|vulkan] [--ctx N]` -- one-shot
    raw completion (no chat template, except `--edit-file` below) with separate prefill/decode timing.
    - `-n N` is optional: without it the reply budget is the room left in `n_ctx` after the
      prompt minus a 64-token EOS margin; an explicit `-n` that no longer fits is clamped to the
      remaining room with a warning. Decoding stops early at EOS / the model's end-of-turn
      token, and `--edit-file` additionally stops as soon as the rewrite's closing ``` fence has
      been emitted.
    - `--edit-file <path> [--edit-on]`: wraps `<path>`'s contents plus the `-p` instruction into
      a whole-file-rewrite prompt (e.g. `-p "Add debugging" --edit-file main.c --edit-on`),
      encoded with the model's own chat template -- this is an instruction-following task, and a
      bare raw prompt makes instruct models either emit EOS immediately (nemotron_h: 0 decoded
      tokens) or degenerate into a bare-fence loop (llama-3.2-3b). `--thinking`/`--no-thinking`
      control the reasoning trace for this mode (default: off; a think trace costs minutes at
      these decode speeds).
      Without `--edit-on` it's a dry run (prints the rewrite, touches nothing). With
      `--edit-on`, the original is first backed up to `<path>.original`, then `<path>` is
      overwritten -- but only if the output passes a basic sanity check (non-trivial length,
      non-fence/substantive content, balanced braces/parens/brackets), since these are small
      models with no diff/code-edit fine-tuning and can truncate or hallucinate on a full-file
      rewrite.
      Extraction is robust to common small-model output patterns: it takes the LAST non-empty
      ``` fence (so an "improved" rewrite that comes after a "simple" draft is preferred), and
      also recognizes a `===== BEGIN FILE =====` / `===== END FILE =====` answer format that
      some models produce after seeing those markers in the prompt.
    - `--search [--max-searches N]`: multi-step web-search tool call (off by default). The prompt
      tells the model to emit `[SEARCH: <query>]` when it needs current information; the CLI
      pauses decoding, runs the search with `curl | python3` (DuckDuckGo html endpoint, HTML
      stripped, ~15 s cap), prefills a `[SEARCH RESULTS for <query>]: ...` block (or
      `[SEARCH FAILED for <query>]` when nothing came back) into the live context, and resumes
      decoding -- so the model can chain think -> search -> think -> search -> answer. Hard-capped
      at N rounds per prompt (default 3; `--max-searches` implies `--search`); each result is
      truncated to ~2000 chars, further if `n_ctx` demands it. Each round prints
      `[Web Search i/N]: <query>` to stderr while it runs.
  - `ggml-loader chat <model.gguf> [--system "text"] [-n N] [--backend cpu|vulkan]` -- interactive
    multi-turn chat using the model's actual instruct format (phi3-style, ChatML, or
    llama3's header-id format, picked automatically), conversation history threaded through the
    KV cache -- each turn only encodes+prefills the newest message, not the whole conversation.
    - `-n N`: optional max tokens per reply (a reasoning model's thinking tokens count towards
      it, so raise it when replies get cut off). Default: the room left in the KV cache after
      the prompt minus a 64-token EOS margin; an explicit `-n` that exceeds the room is clamped
      to it with a warning. The available room shrinks as the conversation grows.
    - `--show-thinking`/`--hide-thinking`: a reasoning model's think blocks are shown in dim gray
      by default; `--hide-thinking` collapses them to a `[thinking hidden]` note instead.
      `--thinking`/`--no-thinking` control whether the model reasons at all (default: on).
    - `--search [--max-searches N]`: multi-step web-search tool call (off by default; same
      contract as `run` above). The system prompt tells the model to emit `[SEARCH: <query>]`;
      the CLI pauses generation, runs the search with `curl | python3` (DuckDuckGo html endpoint,
      HTML stripped, ~15 s cap) and feeds a `[SEARCH RESULTS for <query>]: ...` block back into
      the live context (default 3 rounds per reply via `--max-searches`, each result truncated to
      ~2000 chars), so the model can search again after reading the results. A failed search
      injects `[SEARCH FAILED for <query>]` instead, so the model can recover; each round prints
      `[Web Search i/N]: <query>` on stderr.
    - Multi-line input: a pasted block is sent as one message -- bracketed paste when the
      terminal supports it, otherwise a rapid-line burst is detected; `/paste` followed by
      the text and a final `/end` is the explicit form. (A plain multi-line paste would
      otherwise arrive as one message per line, each getting its own full reply.)
    - Local slash commands (intercepted by the CLI, never sent to the model; `/help` lists
      them): `/load-file <path>` attaches a file's contents to the next message (in a
      `File <path> contents:` + `===== BEGIN/END FILE =====` block); `/save [file]` writes the
      conversation as a JSON message array (default `chat_history.json`); `/load <file>`
      replaces the conversation with a saved one and rebuilds the KV cache by re-prefilling
      the loaded history. Any other text that starts with `/` is sent as a normal message.
  - `ggml-loader bench <model.gguf> [--backend cpu|vulkan] [-p n][-n n][-r reps] [--ctx N]` -- averaged
    pp/tg throughput (one untimed warmup cycle, then N timed repetitions on synthetic tokens),
    directly comparable to `llama-bench`'s own `-p`/`-n`/`-r`.

Verified against Phi-4-mini-instruct, Qwen2.5-Coder-7B, Qwen3-0.6B, and Llama-3.1-8B-Instruct,
all Q4_K_M/Q8_0 quantized.

`<model.gguf>` above also accepts an Ollama model reference (e.g. `qwen2.5-coder:7b-instruct-q4_K_M`)
for anything already pulled with `ollama pull <name>` -- it's resolved to the underlying blob under
`~/.ollama/models` (or `$OLLAMA_MODELS`) via the pull's manifest, since that blob is byte-identical
GGUF this loader already reads directly, no re-export/copy needed. Falls straight through to the
normal "failed to load" error if the argument is neither an existing file nor a resolvable Ollama
name.

### SDXL / image diffusion -- beta, CPU-only, not the project's focus

A basic SDXL Turbo text-to-image path exists (`clip-encode`, `unet-denoise`, `vae-decode`, and
`txt2img` which chains all three): dual CLIP text encoder, the full U-Net, VAE decoder, and a
single-step Euler sampler (SDXL Turbo's native 1-step mode -- no CFG, no negative prompt, no
multi-step schedule). `txt2img` loads and frees each stage's weights in turn rather than holding
CLIP+UNet+VAE resident all at once, which is what keeps 512x512 from OOMing on-device.

This was built as a side track to prove the loader could drive something other than a text LLM,
not because SDXL is this project's actual goal (that's still the Mali GPU backend, see below/
"Findings"). It's CPU-only, single-step-Turbo-only, and not expected to see much further
investment -- no multi-step sampler, no CFG/negative prompts, no other SDXL-family architectures
(e.g. base SDXL checkpoints need real multi-step + CFG support to produce anything but noise
through this path today), no Vulkan port planned for it specifically.

## What's NOT here yet

- **A Mali-specific GPU backend.** This is the actual point of the project and hasn't been
  started. See "Findings" for why it's needed rather than just using ggml's existing backends.
- Other architectures the user has test models for but that turned out to be much bigger
  undertakings than "add a few flags": **qwen35** (Qwen3.5+, a hybrid Gated DeltaNet + MoE +
  vision-language architecture -- new op types we don't have at all), **nemotron_h** (NVIDIA's
  hybrid Mamba2+Transformer -- mixed layer types per block, no RoPE), and **dflash** (turned out
  to not be a standalone architecture at all -- it's a block-diffusion *draft model* for
  speculative decoding, needs a whole target+draft pipeline, not just a graph-building function).
- The exact pretokenizer regex each model's `tokenizer.ggml.pre` names (e.g. "gpt-4o", "qwen2",
  "llama-bpe"). The encoder here uses the classic GPT-2 splitting rules instead
  (contractions/letter-run/digit-run/other-run/whitespace-run). Byte-level BPE is lossless
  regardless of where the pretokenizer draws chunk boundaries, so this still produces valid,
  round-trippable tokenization -- it just may not match the official tokenizer's exact token
  boundaries/count in edge cases.
- A prompt encoder path beyond plain text (no multimodal, no tool-calling).

## Findings that shaped the plan

Benchmarked on the actual target hardware (MediaTek Dimensity, Mali-G615 MC6) before writing any
GPU code, rather than assuming:

- ggml/llama.cpp's OpenCL backend is Adreno-only (`GGML_OPENCL_USE_ADRENO_KERNELS`, prebuilt
  Adreno binary kernels) -- Mali isn't mentioned in it at all.
- The Vulkan backend is portable and does run on Mali, but on this GPU llama.cpp's own
  `llama-bench` shows it gives **almost no speedup on prompt processing** over CPU (pp128: 4.62
  CPU vs 4.76 Vulkan tok/s) and a real but modest ~3x on token generation. The GPU reports no
  matrix cores and no native int8 dot product, which is exactly what the generic coopmat-based
  Vulkan kernels are tuned to exploit -- so they're not getting traction here.
- Conclusion: there's a real, unfilled gap for Mali specifically (unlike Adreno, which has a
  hand-tuned path), which is what the planned custom backend is for.

## Benchmarks (informal, one device, no slogan yet)

`ggml-loader bench`, `-p 128 -n 64 -r 5`, vs `llama-bench` with matching `-p`/`-n`/`-r`, same
device, Phi-4-mini-instruct Q4_K_M. `llama-bench` numbers are with `-fa on` (flash attention
made no measurable difference either way -- see below -- these are the fairer of the two runs):

| | pp128 (prefill) | tg64 (decode) |
| --- | --- | --- |
| **ggml-loader, cpu**    | 12.04 ± 1.18 | 3.42 ± 0.85 |
| **ggml-loader, vulkan** | 20.04 ± 0.19 | 10.99 ± 0.06 |
| llama.cpp, cpu (`-ngl 0 -fa on`)   | 4.79 ± 0.02 | 3.47 ± 0.21 |
| llama.cpp, vulkan (`-ngl 99 -fa on`) | 4.81 ± 0.03 | 9.81 ± 0.02 |

On this run: ~4.2x faster than llama.cpp's Vulkan on prefill, ~1.12x faster on decode.
Flash attention was tested and ruled out as the explanation (`-fa on` vs off moved llama.cpp's
numbers by noise-level amounts: pp128 4.76->4.81, tg64 9.80->9.81) -- both this project and
llama.cpp are using ggml's identical underlying Vulkan kernels either way, so the gap isn't
"we wrote faster kernels," it's something else, most likely llama.cpp's more general-purpose
overhead (multi-sequence batching, a more flexible KV cache, speculative-decoding hooks) that
this narrower, single-purpose implementation doesn't pay -- but that's still an unverified
hypothesis, not confirmed the way the `-fa` test was. Re-testing on more than one
architecture/model size is the natural next check before trusting this fully.

## Building

```sh
git submodule update --init --recursive   # pulls third_party/ggml
cmake -S . -B build
cmake --build build --target ggml-loader -j$(nproc)
```

Three small local patches are applied to the vendored `third_party/ggml` (uncommitted, in the
working tree, not upstream):

- Two work around a Termux `shaderc` bug that otherwise fails the Vulkan shader build
  (`shaderc: internal error: ... Invalid capability operand`) on `coopmat2`/`USE_OCP_FP4`
  shaders -- see the comments at the patched sites in `third_party/ggml/src/ggml-vulkan/` for
  details. These shader variants aren't relevant to Mali (no matrix cores) so disabling them
  costs nothing on this hardware.
- One fixes a real heap-buffer-overread in the CPU backend's SVE (128-bit vector length)
  `ggml_vec_dot_q6_K_q8_K` (`third_party/ggml/src/ggml-cpu/arch/arm/quants.c`): it read 32 bytes
  of a 16-byte `block_q6_K.scales` field via a deinterleaving `svld2_s64` load. The extra bytes
  were always discarded before use (so results were never wrong), but the read itself segfaults
  whenever a q6_K tensor's last block happens to end near an unmapped page -- reproduced with a
  real Ollama-pulled `qwen2.5-coder:7b-instruct-q4_K_M` on `--backend cpu` (this device's SVE is
  128-bit). Fixed with a correctly-sized 16-byte load split into its low/high halves instead;
  verified bit-for-bit identical output against `--backend vulkan` (unaffected by the bug) and
  against the other already-working models before/after the patch.

## Usage

```sh
./build/ggml-loader inspect model.gguf
./build/ggml-loader run model.gguf -p "The capital of France is" -n 24 --backend vulkan
./build/ggml-loader run model.gguf -p "Add debugging" --edit-file main.c --edit-on
./build/ggml-loader run qwen2.5-coder:7b-instruct-q4_K_M -p "The capital of France is" --backend vulkan
./build/ggml-loader chat model.gguf -n 2048 --backend vulkan
./build/ggml-loader bench model.gguf -p 128 -n 64 -r 5 --backend vulkan
```

## License

MIT, see [LICENSE](LICENSE).
