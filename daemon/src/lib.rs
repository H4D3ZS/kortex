pub mod gc;
pub mod gist;
pub mod vfs_state;
pub mod crypto;
pub mod symlink;
pub mod watcher;
pub mod chain;
pub mod watermark;
pub mod neural_math;

/// The Cognitive Kernel represents the VFS daemon bridging .aim files and the system.
pub struct CognitiveKernel {
    pub runtime_state: vfs_state::VfsState,
    pub securityLayer: crypto::SecurityLayer,
    pub shadow_watcher: watcher::ShadowWatcher,
    pub chain_vault: chain::QuantumChain,
    pub safety_watchdog: watermark::SoftBindingWatchdog,
}

impl CognitiveKernel {
    pub fn new() -> Self {
        Self {
            runtime_state: vfs_state::VfsState::new(),
            securityLayer: crypto::SecurityLayer::new(),
            shadow_watcher: watcher::ShadowWatcher::new(),
            chain_vault: chain::QuantumChain::new(),
            safety_watchdog: watermark::SoftBindingWatchdog::new(),
        }
    }

    /// Spin-up the background asynchronous VFS handlers and the Garbage Collector
    pub async fn spawn_background_infrastructure(&self) {
        let _gc_handle = tokio::spawn(async move {
            // Initiate the background consolidation loop for Memory Tiering (L1, L2, L3)
            // MemoryGarbageCollector runs on intervals to decay logs into semantic truths.
        });
    }
}

impl Default for CognitiveKernel {
    fn default() -> Self {
        Self::new()
    }
}
