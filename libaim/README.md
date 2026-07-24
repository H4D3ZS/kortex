# libaim — the `.aim` catalog

Compressed, searchable workspace context with just-in-time chunk
inflation. Given a request, it resolves the handful of source excerpts
that actually matter and injects only those.

Measured on this repository: **31,811 files → 243,824 chunks, indexed in
55 s**, retrieval in **3–33 ms**.

## What changed

The previous `.aim` path read a fixed blob of workspace metadata and
stapled a summary of it onto *every* prompt. That costs the same tokens
on every request and gets less useful as the workspace grows. This is
actual retrieval.

## Layout

A catalog is a directory with three files:

| file | contents |
|---|---|
| `catalog.aim` | container: header, chunk table, path heap, gist, IDF table, zstd payload |
| `catalog.tvim` | [turbovec](https://github.com/RyanCodrai/turbovec) `IdMapIndex` — one quantized vector per chunk |
| `catalog.json` | which embedder built it, and with what settings |

turbovec (Google Research's TurboQuant) compresses each 1536-d f32
embedding from 6 KB to 4 bits/coordinate — 768 bytes, an 8× reduction
with no codebook training. Chunk *text* is zstd-compressed separately and
only decompressed on a hit.

The index is a sibling file rather than an embedded section because
turbovec owns its own versioned format and only exposes path-based
read/write; keeping it separate avoids copying index bytes out of the
mmap just to hand turbovec a path.

## Usage

```bash
# Build a catalog (defaults to <workspace>/.aim)
aim-index build /path/to/workspace

# Inspect it
aim-index stats /path/to/workspace/.aim

# Try a query — use this to tune the gate against your own corpus
aim-index query /path/to/workspace/.aim "fix the mailbox IRQ starvation" --show-text

# Restrict to matching paths (roughly 13x faster: turbovec skips whole
# 32-vector blocks that contain no allowed slot)
aim-index query ./.aim "irq starvation" --scope hw/misc/apple_mbox
```

### Dense embeddings

The default embedder is lexical and needs no model or network. For
queries phrased in prose that shares no tokens with the code, use a real
embedding model:

```bash
# Lemonade (default, :13305) — also emulates Ollama's API on the same port
aim-index build . --model nomic-embed-text

# Explicit backend
aim-index build . --backend ollama --server http://localhost:11434 --model nomic-embed-text
```

`--backend auto` (the default) probes Lemonade, then any
OpenAI-compatible endpoint, then Ollama, and uses whichever answers.
Requires `--features http-embed`.

The catalog records which embedder built it. Querying with a different
one is refused rather than silently returning noise — that mismatch is
the single most damaging failure mode in a retrieval system, because
scores stay in range and nothing errors.

## Two findings worth knowing

### Similarity cannot decide *whether* to retrieve

Measured on the real 243k-chunk corpus:

| query | top cosine |
|---|---|
| `thanks` | 0.44 |
| `hello` | 0.38 |
| `fix the AGX mailbox IRQ starvation in apple_mbox.c` | 0.32 |
| `implement zstd decompression for chunk payloads` | 0.16 |

Score is *anti-correlated* with relevance. A one-word query puts all its
mass on one feature, so any chunk containing that word scores near
perfectly, while a long query spreads mass across features no single
chunk contains. **No threshold — absolute or relative — separates these.**

So the decision is made from the query instead (`gate.rs`): a request
needs at least 3 distinct non-conversational tokens, or one unambiguous
code token (a path, a `snake_case` identifier, or a filename with a known
extension). `hello` and `thanks` never reach the index; `apple_mbox.c`
retrieves on its own.

### IDF helps ranking, not gating

IDF weighting is applied and does improve ranking. It does **not** fix
the gating problem, and the reason is counterintuitive: `hello` and
`thanks` are *rare* in a code corpus, so IDF weights them **up**. This
was measured, not assumed.

## Retrieval configuration

`RetrievalConfig` gates on a fraction of the best score
(`relative_floor`, default 0.6) rather than a fixed cosine, because
absolute magnitude tracks query length rather than relevance. The
absolute `fault_threshold` is a low floor for "the corpus has nothing
about this", not a ranking knob.

Note this is a *cosine*, not the 0.85 attention-activation threshold from
the design notes — different quantity, different scale.

## Memory pinning

`HeatMap` tracks which chunks a session keeps retrieving, with
exponential decay so heat reflects the current task. `Catalog::pin_chunks`
locks those payload pages into physical RAM (`VirtualLock` on Windows,
`mlock` on Unix). This is Colibri's routing-heat idea applied to a code
catalog.

Pinning is best-effort: both syscalls are quota-limited, so a refusal is
normal on an untuned machine and is reported rather than raised. An
unpinned catalog is slower on a cold page, never incorrect.

## C ABI

`ffi.rs` exposes the catalog to the VSCodium extension over `ffi-napi`.
The three legacy symbols (`aim_mount_vfs`, `aim_get_tensor`,
`aim_unmount_vfs`) keep their exact signatures. The new
`aim_catalog_open` / `aim_catalog_query` / `aim_catalog_close` return
JSON. Every function is null-safe and catches panics — unwinding across
the FFI boundary would abort the editor.
