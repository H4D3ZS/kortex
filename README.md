# 🧠 KORTEX: Holographic Virtual File System (.aim Neural VFS)

[![License: AGPL-v3](https://img.shields.io/badge/License-AGPL_v3-red.svg)](LICENSE)
[![Rust: Stable](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://rust-lang.org)
[![Technical Report](https://img.shields.io/badge/Research-Technical%20Report-blue.svg)](./Neural_AIM_VFS_A.I_Kontex_Solution.pdf)
[![Affiliation: Cyber Ifrit](https://img.shields.io/badge/Publisher-Cyber%20Ifrit%20Software%20Services-purple.svg)](https://github.com/Cyber-Ifrit)

**Kortex** is a sovereign, high-performance cognitive infrastructure that solves the "Context Inflation" and "VRAM Gentry" crises in agentic AI development. By decoupling massive physical filesystems from the active Large Language Model (LLM) context window, Kortex enables autonomous software agents to command multi-gigabyte repositories with stable $O(1)$ token prefixes.

---

## 📄 Technical Report
The mathematical framework and design rationale behind the Kortex architecture are written up
as a self-published technical report — not peer-reviewed, no venue, no DOI. Read it as a design
document, not a validated publication:

📖 **[Holographic Virtual File Systems: Zero-Token Cognitive Integration for Autonomous LLM Software Agents via Latent Superposition (PDF)](./Neural_AIM_VFS_A.I_Kontex_Solution.pdf)**  
*Author: Rolando H. Ferrer Jr. (Sole Proprietor)*  
*Cyber Ifrit Software Development Services (Technical Report No. CI-2026-01)*

---

## 🎯 What is the Kortex `.aim` Neural VFS?

Traditional AI software agents are severely bounded by raw context injection limitations. When navigating massive repositories containing hundreds of scripts, injecting full-file contents scales token billing costs linearly, disperses model attention coefficients, and invalidates temporary prefix caches at every single keystroke.

**Kortex completely departs from this paradigm by building a dual-layer cognitive architecture:**

```
                        COGNITIVE LAYER
┌─────────────────────────────────────────────────────────────┐
│  L1/L2 Active Memory: 6KB Limbic Gist Vector                │
│  - 1,536-dimensional float32 vector in VRAM/local memory    │
│  - Holographic Key-Value Superposition (see capacity below) │
│  - Remains resident permanently for stable O(1) prefix hit  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               │ Page Fault Trigger (Sim > 0.85)
                               ▼
                        PHYSICAL LAYER
┌─────────────────────────────────────────────────────────────┐
│  L3 Structural Catalog: 20MB Memory-Mapped .aim DB          │
│  - Persistent Merkle-DAG hashes and chunk directories        │
│  - Just-in-Time File Inflation (JIT) on local target drive   │
│  - High-performance, zero-copy io_uring stream buffers      │
└─────────────────────────────────────────────────────────────┘
```

When the AI core experiences a "Page Fault" (detecting a file reference with direct relevance coordinates), Kortex automatically inflates and loads the specific required code snippet from the **L3 Disk catalog (.aim)** into active focus using a low-latency binary projection loop, completely bypassing redundant model billing.

---

## 🧬 The Mathematical Core: Superposition Bindings

Kortex relies on **Holographic Reduced Representations (HRR)** and **Vector Symbolic Architectures (VSA)**. Our research exposes and corrects a critical mathematical flaw present in early VFS indexing algorithms: **Cascade Circular Convolution Decay**.

### The Cascade Decay Problem (Why Serial Paging Fails)
If script chunks are bound consecutively in a sequential circular convolution loop:

$$\mathbf{v}_N = \mathbf{v}_0 \circledast \mathbf{c}_1 \circledast \mathbf{c}_2 \dots \circledast \mathbf{c}_N$$

The spectral frequency components undergo exponential phase polarization. Under repeated convolving without continuous re-normalization, the signal vector collapses rapidly to zero:

$$\lim_{N \to \infty} \mathbb{E}\left[ \langle \mathbf{v}_N, \mathbf{c}_i \rangle \right] = 0, \quad \forall i \in \{1,\dots,N\}$$

This turns the persistent index vector into high-dimensional isotropic white noise, rendering directory traversal and semantic search mathematically impossible.

### The Kortex Solution: Unitary Path-Key Superposition
Kortex solves this signal decay through **Key-Value Superposition Binding**:

1. For each script file path string, we generate a deterministic **Path Key**: SHA3-256 of the
   path string seeds the construction of a *unitary* HRR vector — one whose Discrete Fourier
   Transform has magnitude exactly 1 at every frequency bin, with a random phase per bin
   (subject to the conjugate symmetry a real signal requires). Concretely: build that spectrum,
   inverse-FFT it, done — see `daemon::neural_math::{path_key, unitary_vector}` for the exact
   code (`kortex/daemon/src/neural_math.rs`).

   An earlier version derived the key directly from the path's own bytes at each vector index
   ($k_i = \sin(\text{Byte}_{(i \bmod L)} \cdot \sin(i))$). Two problems with that: (a) paths
   sharing a prefix (`src/foo/a.rs`, `src/foo/b.rs`) produced strongly correlated keys — exactly
   the files most likely to be retrieved together, and exactly where HRR needs keys to be close
   to orthogonal; hashing the path first fixes this (avalanche effect: one differing byte flips
   ~half the hash's output bits). (b) A vector normalized to unit *length* is not the same as a
   vector with unit magnitude at every *frequency* — the former still has "loud" and "quiet"
   frequency bins, and correlating with such a key does not cleanly invert convolution even for
   a single item with zero interference from anything else. Verified empirically: a
   length-normalized-but-not-unitary key round-trips one bind/unbind at cosine similarity
   ≈0.70; a unitary key round-trips at ≈1.00 (`daemon/examples/hrr_sanity.rs`).

2. The unitary key is convolved with the target chunk's LLM embedding:

   $$\mathbf{v}_{\text{bound}} = \mathbf{k}_{\text{path}} \circledast \mathbf{c}_{\text{embedding}}$$

3. The bound pairs are aggregated using **linear vector superposition** combined with **Test-Time Training (TTT)** weight blending:

   $$\mathbf{v}_{\text{global}}^{(k)} = (1-\alpha) \mathbf{v}_{\text{global}}^{(k-1)} + \alpha \mathbf{v}_{\text{bound}}$$

By performing circular correlation with a target path key, Kortex recovers the script context:

$$\mathbf{b}'_m = \mathbf{v}_{\text{global}} \oplus \mathbf{a}_m$$

### Measured Capacity — Not the SNR Formula's $d/(k-1)$
An earlier version of this section claimed the classic HRR result
$\text{SNR} \approx d/(k-1) \gg 1$ holds "even with 30,000 files superposed." That formula is the
standard *auto-associative cleanup* capacity result (Plate, 1995) — decoding a noisy recovered
vector by nearest-neighbor match against a **known candidate set**. It is not a direct prediction
of raw cosine similarity against a **fixed absolute threshold**, which is what this system
actually uses to decide whether to page a file in (`Sim > 0.85`, above). We measured the real
number instead of assuming the formula transfers.

Why it doesn't transfer: for a *unitary* key (the fix above — needed for exact single-item
recall), correlating the global vector with the wrong item's key doesn't return a
small, spread-out noise term — it returns a full-magnitude, randomly phase-rotated copy of that
*other* item's own content vector, because a unitary key has magnitude exactly 1 at every
frequency, so it can't attenuate anything. Each additional superposed item therefore contributes
noise on the *same scale* as the signal itself, not noise damped by $1/\sqrt{d}$. Measured
accuracy falls off close to $1/\sqrt{k}$ starting from the very first added item — not staying
flat until some large $k$.

| Chunks ($k$) | Mean cos-sim | Retrieval Accuracy (Sim > 0.85) |
|:---:|:---:|:---:|
| 1 | 1.000 | 100.0% |
| 2 | 0.715 | 0.0% |
| 5 | 0.440 | 0.0% |
| 10 | 0.311 | 0.0% |
| 20 | 0.219 | 0.0% |
| 100 | 0.102 | 0.0% |
| 2,000 | 0.022 | 0.0% |
| 20,000 | 0.005 | 0.0% |
| 30,000 | 0.007 | 0.0% |

Run it yourself: `cargo run --release --example hrr_benchmark -p daemon`
(`daemon/examples/hrr_benchmark.rs`). It's the same `neural_math` code the app ships with, not a
separate calculation.

**What this means in practice:** a single 1536-dim global vector reliably holds *one* superposed
item under this system's own retrieval threshold. Holding more needs one of: (a) a materially
looser decode rule — nearest-neighbor against a known candidate set, the regime the $d/(k-1)$
result actually describes, cheap to add but changes what "retrieval" means; or (b) partitioning
the index instead of superposing everything into one vector — many smaller vectors (one per
file, cluster, or directory) rather than one global 6KB vector. `LimbicIndex`/`LimbicMap` in
`hades-kernel/src/jit_decompression/semantic_map.rs` already scaffold exactly this
(per-cluster indices, activation-threshold retrieval) — it just isn't wired into the live
chunk-indexing path yet (`neuraldrive/src-tauri/src/lib.rs` currently accumulates everything
into one `global_vector`). That's the real next step, not yet shipped, and this README no
longer claims 30,000-file-scale superposition works until it's been measured at that scale
under that mechanism.

---

## ⚡ Real-World Benchmarks

Semantic retrieval accuracy is covered above ("Measured Capacity"), with a runnable harness —
that's the one number in this README with a script behind it.

**Prompt-cache hit rate, latency reduction, and token-cost reduction are not covered by a
harness yet.** An earlier version of this README stated specific figures for these (a cache hit
rate, a latency reduction, a token-cost reduction) without one. Removed rather than left in:
they'd need an actual measured session against a real model server (prefix-cache hit/miss
counts, wall-clock latency, token counts, before/after) to say honestly, and that harness
doesn't exist yet. The mechanism this claims to help — KV-slot prefix caching — is real and
shipped (see the outer IDE's `kortex/aim-proxy` integration); the *numbers* for it are not, until
measured.

---

## 🏗 Kortex Repository Architecture

*   **`aim-proxy/`**: Highly parallel Rust proxy layer (port `1536`) implementing MitM interception for Anthropic, OpenAI, and Ollama APIs. Automatically extracts local `.aim` catalogs to inject optimized system prefixes.
*   **`hades-kernel/`**: High-performance Rust substrate handling memory-mapped I/O, zero-copy mmap buffers, and digital signatures.
*   **`neuraldrive/`**: 3D semantic graph mapping visualizer GUI written in React and Tauri to monitor superposition vectors interactively.
*   **`daemon/`**: Visual mapping engine using clip/siglip indices.

---

## 🚀 Getting Started

### 1. High-Performance Build
```bash
# Build the Rust MITM proxy
cd kortex/aim-proxy
cargo build --release --bin aim-proxy

# Build the background daemon mapping engine
cd ../daemon
cargo build --release
```

### 2. Running the Proxy
Kortex integrates natively as a persistent background daemon. To connect Cursor or Cursor-like agents:
1. Execute the proxy:
   ```bash
   ./target/release/aim-proxy.exe
   ```
2. Re-route your AI agent's base URL API endpoint to:
   ```text
   http://127.0.0.1:1536/v1
   ```
The proxy will automatically detect active code modifications, compile changes into the L3 `.aim` catalog, update the L1/L2 Limbic Vector, and append the cached prefix to all outbound developer messages.

---

## 🤝 Reference Publications in this Architecture

1. **Anthropic.** (2024). *Introducing prompt caching on the Anthropic API*. Anthropic Research Blog.
2. **Gu, A., & Dao, T.** (2023). *Mamba: Linear-time sequence modeling with selective state spaces*. arXiv:2312.00752.
3. **Schlegel, K., Neubert, P., & Protzel, P.** (2022). A comparison of Vector Symbolic Architectures. *Artificial Intelligence Review*, 55(6), 4523--4555. [doi.org/10.1007/s10462-021-10110-3](https://doi.org/10.1007/s10462-021-10110-3)
4. **Sun, Y., Liu, Z., Kirschstein, L., Efros, A. A., & Wang, X.** (2024). Learning to filter context with test-time training. arXiv:2407.04621. [arxiv.org/abs/2407.04621](https://arxiv.org/abs/2407.04621)

---

## 🛠 Development & Attribution

**Human-led, AI-assisted.** The architecture, research, and every design decision here are
human work. The author planned the system, did the research, found the approaches, decided
what the code should do and how it should read, and reviewed and cleaned everything that
landed. AI was used as a **boilerplate generator and coding assistant** — turning a decided
design into a first draft of code — under human direction and review. Nothing here is
"AI-generated software"; it is engineered by a human who used AI as a tool.

**Combined stack & credits.** This repository ships with **[ROCmFPX](ROCmFPX/)** as a git
submodule — the AMD RDNA4 inference engine by **Carlo (`charlie12345`)**, a fork of
llama.cpp/ggml (MIT). All GPU-side capability (low-bit `Q*_ROCMFPX` quants, ROCmFP4/NVFP4,
MTP / ngram / EAGLE-3 / DFlash speculative decoding, KV slot save/restore) is Carlo's work.
kortex itself is AGPL-3.0. See **[NOTICE.md](NOTICE.md)** for the full licensing and credits.

---

**Built by the Sovereign Systems Architect under the Cyber Ifrit Software Development Services ecosystem.**  
*"The best GPU is the one you already have. Make it infinite."*
