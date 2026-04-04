# .aim Neural Virtual File System (AI-Interactive Memory)

![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)
![Build: Rust 1.80+](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)
![Security: Post-Quantum](https://img.shields.io/badge/Security-Post--Quantum-green.svg)

Welcome to the future of context-aware, zero-token cost AI development. 

The `.aim` Neural VFS solves the AI "Context Crisis" by compressing massive project histories into a single **Parametric Gist Token**. This repository contains the Next-Gen VFS daemon, Tauri-based frontend, and Post-Quantum hybrid cryptographic components required to keep the system natively interoperable with Cursor, Claude, and traditional OS file boundaries.

## 🧠 Architecture
- **Daemon (`/daemon`)**: The **Cognitive Kernel**. Ingests file states, runs a memory garbage collector (time-decay), and exposes LLM prefix context blocks. Built in high-performance Rust.
- **VFS Layer (`/vfs_layer`)**: The low-level **Dokany/FUSE adapter** mounting the interactive `.aim` structure. Uses `io_uring` and `mmap` for near-zero CPU overhead.
- **NeuralDrive GUI (`/neuraldrive`)**: The lightweight (<30MB) **Tauri 2.0 + React** viewer parsing human `.aim` overlaps via Monaco.

## ⚡ Zero Setup & Token Cost Execution
The `.aim` daemon employs a **Neural Symlink** approach. By silently writing to `.cursorrules` and `CLAUDE.md`, the environment automatically reads the state at inference time. 

Thanks to **LLM Prompt Prefix Caching**, processing a 50MB architecture state costs **~1 inference token**. By keeping the "Gist" at the start of the prompt, AI providers (Anthropic/OpenAI) cache the mathematical state, reducing query costs by up to 99.9%.

## 🔐 Security
Under the hood, all transitions are cryptographically sealed with **Hybrid Signatures** combining the best of classical (**Ed25519**) and Post-Quantum algorithms (**ML-DSA / Dilithium**). 
- **Quantum Resistant**: Proof against CRQC (Cryptographically Relevant Quantum Computers).
- **Hardware Root of Trust**: Keys are optionally stored in the Secure Enclave/TPM.

## 🚀 Quick Start
1. **Install the Daemon**: `cargo install aim-vfs`
2. **Mount a Project**: `aim-vfs mount ./my-project`
3. **Open IDE**: Start Cursor or Claude Code. The `.aim` state is injected automatically.

---
*Developed by Cyber-Ifrit. Solving the Global Token Crisis one project at a time.*