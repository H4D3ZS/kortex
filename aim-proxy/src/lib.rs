//! aim-proxy as a library, so the retrieval proxy can be hosted **in-process**
//! by the IDE (`vscodium-rust/src-tauri`) as well as run as the standalone
//! `aim-proxy` binary. The binary (`main.rs`) is a thin wrapper over
//! [`server::build_router`].

pub mod api_manifest;
pub mod retrieval;
pub mod server;
