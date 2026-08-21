# NOTICE — licensing & attribution

This repository combines two independently-licensed works. Each keeps its own
license; the whole is an **aggregate**, not a relicensing of either part.

## kortex — AGPL-3.0
Copyright © Rolando H. Ferrer Jr. (`H4D3ZS`), Cyber Ifrit Software Development Services.
Everything in this repository **except the `ROCmFPX/` submodule** is licensed under the
GNU Affero General Public License v3.0 — see `LICENSE`.

kortex is the Holographic VFS / Neural AIM repo-memory layer: `aim-proxy` (context
injection), `aim-index` (catalog builder), hybrid dense + BM25 + structural retrieval,
`aim-mcp` (tool server), and `turbovec` (the bundled quantized vector index).

## ROCmFPX/ (git submodule) — MIT
The `ROCmFPX/` submodule is a fork (`H4D3ZS/ROCmFPX`) of **ROCmFPX by Carlo (`charlie12345`)**
— https://github.com/charlie12345/ROCmFPX — itself a fork of **llama.cpp / ggml**.
It is licensed under the **MIT License** ("Copyright (c) 2023-2026 The ggml authors"),
which governs that entire subtree. See `ROCmFPX/LICENSE`.

**All GPU-side capability this project runs on is Carlo's work**: the `Q*_ROCMFPX`
low-bit AMD quant formats, ROCmFP4 / native NVFP4, and speculative decoding on RDNA4
(MTP, `ngram-map-k`, EAGLE-3, DFlash / DSpark), plus KV slot save/restore and Vulkan/HIP
for gfx1200. **Thank you, Carlo, and the ggml/llama.cpp authors.**

### Modifications in this fork
- `convert_hf_to_gguf.py` — maps DFlash-2 draft tensors (candidate selector + per-layer
  grouped dynamic convolutions) so a DFlash-2 draft converts. Intended upstream to ROCmFPX
  as a PR. Reference: z-lab/dflash (https://github.com/z-lab/dflash). These modifications
  to MIT-licensed files remain MIT.

## AGPL × MIT — how they coexist
MIT is permissive and compatible with AGPL: MIT-licensed code (the ROCmFPX submodule) may be
distributed alongside AGPL code. The submodule stays MIT; kortex stays AGPL-3.0. If you
convey the combined work over a network, AGPL-3.0 §13 applies to the kortex portion only —
the MIT portion carries no such obligation.

## Other components credited where used
- **Sharp chat template** — froggeric / `peculiar-ragdoll` (HF `peculiar-ragdoll/Qwen-Sharp-Chat-Templates`).
- **DFlash / DFlash-2** — z-lab (https://github.com/z-lab/dflash).
- **Base model** — Qwen3.8-27B, Alibaba Qwen (Apache-2.0).
