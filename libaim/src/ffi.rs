//! C ABI, consumed by the VSCodium extension over `ffi-napi`.
//!
//! Two generations live here:
//!
//! * **Legacy** — [`aim_mount_vfs`], [`aim_get_tensor`], [`aim_unmount_vfs`].
//!   These map a single-tensor `.aim` file and hand back a pointer to
//!   its 1536 floats. `vscodium-extension` binds these three symbols by
//!   name, so their signatures and behaviour are frozen.
//! * **Catalog** — [`aim_catalog_open`], [`aim_catalog_query`], and
//!   friends. These expose actual retrieval: hand in a query string, get
//!   back JSON describing which chunks faulted in and their text.
//!
//! # Conventions
//!
//! Every function is null-safe and returns null (or a negative count) on
//! failure rather than unwinding — a panic across the FFI boundary would
//! abort the whole editor process. Every string returned by this module
//! is owned by the caller and must be released with
//! [`aim_string_free`]; every handle with the matching `*_close`.

use std::ffi::{c_char, c_int, CStr, CString};
use std::fs::File;
use std::path::Path;
use std::sync::Mutex;

use memmap2::Mmap;

use crate::catalog::{Catalog, RetrievalConfig};
use crate::embed::{Embedder, HashEmbedder};
use crate::heat::HeatMap;

// ---------------------------------------------------------------------
// Legacy single-tensor API — signatures frozen for vscodium-extension
// ---------------------------------------------------------------------

/// Opaque handle to a memory-mapped legacy `.aim` file.
pub struct AimMemory {
    _file: File,
    mmap: Mmap,
}

/// Map `<path>/.aim/memory.aim` and return an opaque handle, or null.
///
/// # Safety
///
/// `path_ptr` must be null or a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn aim_mount_vfs(path_ptr: *const c_char) -> *mut AimMemory {
    if path_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let Ok(path_str) = CStr::from_ptr(path_ptr).to_str() else {
        return std::ptr::null_mut();
    };

    // Built with `Path::join` rather than a `\`-formatted string so the
    // same call works on macOS, which is now the primary dev platform.
    let target = Path::new(path_str).join(".aim").join("memory.aim");

    let Ok(file) = File::open(&target) else {
        return std::ptr::null_mut();
    };
    let Ok(mmap) = Mmap::map(&file) else {
        return std::ptr::null_mut();
    };

    Box::into_raw(Box::new(AimMemory { _file: file, mmap }))
}

/// Return a pointer to the legacy 1536-float gist tensor and write its
/// element count to `out_size`. Returns null if the file is too small.
///
/// # Safety
///
/// `aim_ptr` must come from [`aim_mount_vfs`] and still be live.
/// `out_size` must be a valid `size_t*`. The returned pointer borrows
/// the mapping and is invalid after [`aim_unmount_vfs`].
#[no_mangle]
pub unsafe extern "C" fn aim_get_tensor(
    aim_ptr: *mut AimMemory,
    out_size: *mut usize,
) -> *const f32 {
    if aim_ptr.is_null() || out_size.is_null() {
        return std::ptr::null();
    }
    let aim = &*aim_ptr;
    let bytes = &aim.mmap[..];

    // Legacy layout: a JSON header, then the tensor. The header ends at
    // the first '}'. Kept byte-for-byte compatible with the original
    // implementation because the extension depends on this offset rule.
    let Some(brace) = bytes.iter().position(|&b| b == b'}') else {
        return std::ptr::null();
    };
    let header_end = brace + 1;

    const DIM: usize = crate::format::DEFAULT_DIM;
    if header_end + DIM * 4 > bytes.len() {
        return std::ptr::null();
    }

    *out_size = DIM;
    // The tensor offset is whatever follows the header, so it carries no
    // alignment guarantee. Callers read it as raw bytes through
    // `ffi-napi`, which is why this remains a byte pointer cast rather
    // than a Rust `&[f32]` — constructing that slice would be UB.
    bytes[header_end..].as_ptr() as *const f32
}

/// Release a handle from [`aim_mount_vfs`].
///
/// # Safety
///
/// `aim_ptr` must come from [`aim_mount_vfs`] and not be used again.
#[no_mangle]
pub unsafe extern "C" fn aim_unmount_vfs(aim_ptr: *mut AimMemory) {
    if !aim_ptr.is_null() {
        drop(Box::from_raw(aim_ptr));
    }
}

// ---------------------------------------------------------------------
// Catalog retrieval API
// ---------------------------------------------------------------------

/// Opaque handle to an open catalog plus its query-time state.
pub struct AimCatalog {
    catalog: Catalog,
    embedder: HashEmbedder,
    heat: Mutex<HeatMap>,
}

/// Open the catalog directory at `dir_ptr`. Returns null on failure.
///
/// # Safety
///
/// `dir_ptr` must be null or a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn aim_catalog_open(dir_ptr: *const c_char) -> *mut AimCatalog {
    if dir_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let Ok(dir) = CStr::from_ptr(dir_ptr).to_str() else {
        return std::ptr::null_mut();
    };

    // `catch_unwind` because an abort here takes the editor with it.
    let opened = std::panic::catch_unwind(|| Catalog::open(dir));
    let Ok(Ok(catalog)) = opened else {
        return std::ptr::null_mut();
    };

    if catalog
        .check_embedder(&HashEmbedder::new(catalog.dim()).id())
        .is_err()
    {
        // Built by a different embedder; every score would be noise.
        return std::ptr::null_mut();
    }
    let embedder = catalog.query_embedder();

    Box::into_raw(Box::new(AimCatalog {
        catalog,
        embedder,
        heat: Mutex::new(HeatMap::default()),
    }))
}

/// Number of chunks in the catalog, or -1 on a null handle.
///
/// # Safety
///
/// `handle` must come from [`aim_catalog_open`] and still be live.
#[no_mangle]
pub unsafe extern "C" fn aim_catalog_len(handle: *const AimCatalog) -> c_int {
    if handle.is_null() {
        return -1;
    }
    (*handle).catalog.len().min(c_int::MAX as usize) as c_int
}

/// Catalog metadata as a JSON string. Free with [`aim_string_free`].
///
/// # Safety
///
/// `handle` must come from [`aim_catalog_open`] and still be live.
#[no_mangle]
pub unsafe extern "C" fn aim_catalog_meta_json(handle: *const AimCatalog) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    match serde_json::to_string((*handle).catalog.meta()) {
        Ok(s) => into_c_string(s),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Retrieve context for `query_ptr`.
///
/// Returns a JSON object — `{ faulted, best_score, gist_score,
/// injected_tokens, hits: [{ chunk_id, path, line_start, line_end,
/// score, text }], context }` — or null on failure. Free the result with
/// [`aim_string_free`].
///
/// `threshold` below or equal to zero falls back to the default. Set
/// `max_chunks` to zero for the default.
///
/// # Safety
///
/// `handle` must come from [`aim_catalog_open`] and still be live;
/// `query_ptr` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn aim_catalog_query(
    handle: *const AimCatalog,
    query_ptr: *const c_char,
    threshold: f32,
    max_chunks: c_int,
) -> *mut c_char {
    if handle.is_null() || query_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let Ok(query) = CStr::from_ptr(query_ptr).to_str() else {
        return std::ptr::null_mut();
    };
    let handle = &*handle;

    let mut cfg = RetrievalConfig::default();
    if threshold > 0.0 {
        cfg.fault_threshold = threshold;
    }
    if max_chunks > 0 {
        cfg.max_chunks = max_chunks as usize;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let vector = handle.embedder.embed(query)?;
        handle.catalog.page_fault(&vector, &cfg)
    }));

    let Ok(Ok(fault)) = result else {
        return std::ptr::null_mut();
    };

    if let Ok(mut heat) = handle.heat.lock() {
        heat.record_query(fault.hits.iter().map(|h| h.chunk_id));
    }

    let payload = serde_json::json!({
        "faulted": fault.faulted(),
        "best_score": fault.best_score,
        "gist_score": fault.gist_score,
        "injected_tokens": fault.injected_tokens(),
        "dropped_to_budget": fault.dropped_to_budget,
        "hits": fault.hits.iter().map(|h| serde_json::json!({
            "chunk_id": h.chunk_id,
            "path": h.path,
            "line_start": h.line_start,
            "line_end": h.line_end,
            "score": h.score,
            "token_estimate": h.token_estimate,
            "text": h.text,
        })).collect::<Vec<_>>(),
        "context": fault.render_context(),
    });

    match serde_json::to_string(&payload) {
        Ok(s) => into_c_string(s),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Pin the hottest chunks' payload pages into physical RAM, up to
/// `budget_bytes`. Returns a JSON [`crate::PinReport`]-shaped object, or
/// null on failure. Free with [`aim_string_free`].
///
/// # Safety
///
/// `handle` must come from [`aim_catalog_open`] and still be live.
#[no_mangle]
pub unsafe extern "C" fn aim_catalog_pin_hot(
    handle: *const AimCatalog,
    budget_bytes: usize,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    let handle = &*handle;
    let hottest: Vec<u64> = match handle.heat.lock() {
        Ok(h) => h.hottest(256).into_iter().map(|(id, _)| id).collect(),
        Err(_) => return std::ptr::null_mut(),
    };

    let report = handle.catalog.pin_chunks(&hottest, budget_bytes);
    let payload = serde_json::json!({
        "requested": report.requested,
        "pinned": report.pinned,
        "pinned_bytes": report.pinned_bytes,
        "complete": report.complete(),
        "failures": report.failures,
    });
    match serde_json::to_string(&payload) {
        Ok(s) => into_c_string(s),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Release a handle from [`aim_catalog_open`].
///
/// # Safety
///
/// `handle` must come from [`aim_catalog_open`] and not be used again.
#[no_mangle]
pub unsafe extern "C" fn aim_catalog_close(handle: *mut AimCatalog) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Free a string returned by any function in this module.
///
/// # Safety
///
/// `ptr` must come from this module and not be used again.
#[no_mangle]
pub unsafe extern "C" fn aim_string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Move a Rust `String` into a caller-owned C string.
///
/// Interior NUL bytes cannot appear here — every producer is
/// `serde_json`, which escapes them — but the fallible path is handled
/// rather than unwrapped so a future caller cannot turn it into a panic
/// across the ABI boundary.
fn into_c_string(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}
