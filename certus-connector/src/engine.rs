//! EngineInner — wires the Certus component stack and implements the
//! operations exposed by the CertusEngine PyO3 class.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use component_core::query_interface;
use interfaces::{
    CacheKey, DispatcherConfig, DmaAllocFn, DmaBuffer, FormatParams, IBlockDevice,
    IBlockDeviceAdmin, IDispatchMap, IDispatcher, IExtentManager, IGpuServices, ILogger, IpcHandle,
    LookupResult, PciAddress,
};

use crate::keys;

// ─── Transfer job tracking ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum JobKind {
    Store,
    Load,
}

#[allow(dead_code)]
struct TransferJob {
    kind: JobKind,
    keys: Vec<CacheKey>,
    gpu_block_ids: Vec<u64>,
    completed: AtomicBool,
    success: AtomicBool,
}

fn parse_pci_addr(s: &str) -> Result<PciAddress, String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return Err(format!("expected domain:bus:dev.func, got '{s}'"));
    }
    let domain = u32::from_str_radix(parts[0], 16).map_err(|_| format!("invalid domain '{}'", parts[0]))?;
    let bus = u8::from_str_radix(parts[1], 16).map_err(|_| format!("invalid bus '{}'", parts[1]))?;
    let dev_func: Vec<&str> = parts[2].split('.').collect();
    if dev_func.len() != 2 {
        return Err(format!("invalid dev.func '{}'", parts[2]));
    }
    let dev = u8::from_str_radix(dev_func[0], 16).map_err(|_| format!("invalid dev '{}'", dev_func[0]))?;
    let func = u8::from_str_radix(dev_func[1], 16).map_err(|_| format!("invalid func '{}'", dev_func[1]))?;
    Ok(PciAddress { domain, bus, dev, func })
}

// ─── EngineInner ───────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct EngineInner {
    dispatcher: Arc<dyn IDispatcher + Send + Sync>,
    dispatch_map: Arc<dyn IDispatchMap + Send + Sync>,
    gpu_services: Arc<dyn IGpuServices + Send + Sync>,
    gpu_block_size: u64,
    max_cache_entries: usize,
    eviction_watermark: usize,
    entry_count: AtomicU64,
    jobs: Mutex<HashMap<u64, Arc<TransferJob>>>,
    next_internal_id: AtomicU64,
    initialized: AtomicBool,
}

impl EngineInner {
    /// Construct from a Python config dict.
    ///
    /// Instantiates and wires all Certus components:
    /// - SPDKEnvComponent (environment init)
    /// - GpuServicesComponentV0 (CUDA init)
    /// - DispatchMapComponentV0 (key→location index)
    /// - DispatcherComponentV0 (orchestration)
    pub fn from_config(config: &Bound<'_, PyDict>) -> PyResult<Self> {
        let data_pci_addrs: Vec<String> = config
            .get_item("data_pci_addrs")?
            .ok_or_else(|| PyRuntimeError::new_err("missing 'data_pci_addrs'"))?
            .extract()?;

        let metadata_pci_addr: String = config
            .get_item("metadata_pci_addr")?
            .ok_or_else(|| PyRuntimeError::new_err("missing 'metadata_pci_addr'"))?
            .extract()?;

        let gpu_block_size: u64 = config
            .get_item("gpu_block_size")?
            .ok_or_else(|| PyRuntimeError::new_err("missing 'gpu_block_size'"))?
            .extract()?;

        let slab_size_bytes: u64 = config
            .get_item("slab_size_bytes")?
            .and_then(|v| v.extract().ok())
            .unwrap_or(131072);

        let dram_cache_bytes: u64 = config
            .get_item("dram_cache_bytes")?
            .and_then(|v| v.extract().ok())
            .unwrap_or(0);

        let eviction_threshold: f64 = config
            .get_item("eviction_threshold")?
            .and_then(|v| v.extract().ok())
            .unwrap_or(0.8);

        let max_cache_entries: usize = if slab_size_bytes > 0 && dram_cache_bytes > 0 {
            (dram_cache_bytes / slab_size_bytes) as usize
        } else {
            10000
        };

        // --- Initialize SPDK environment ---
        let spdk_comp = spdk_env::SPDKEnvComponent::new_default();
        let spdk_iface = query_interface!(spdk_comp, spdk_env::ISPDKEnv)
            .ok_or_else(|| PyRuntimeError::new_err("failed to query ISPDKEnv"))?;
        spdk_iface
            .init()
            .map_err(|e| PyRuntimeError::new_err(format!("SPDK init failed: {e}")))?;

        // --- Create logger ---
        let log_comp = logger::LoggerComponentV1::new_default();
        let log: Arc<dyn ILogger + Send + Sync> = query_interface!(log_comp, ILogger)
            .ok_or_else(|| PyRuntimeError::new_err("failed to query ILogger"))?;

        // --- Initialize GPU services ---
        let gpu_comp = gpu_services::GpuServicesComponentV0::new_default();
        let gpu: Arc<dyn IGpuServices + Send + Sync> = query_interface!(gpu_comp, IGpuServices)
            .ok_or_else(|| PyRuntimeError::new_err("failed to query IGpuServices"))?;
        gpu.initialize()
            .map_err(|e| PyRuntimeError::new_err(format!("GPU init failed: {e}")))?;

        // --- Create metadata block device ---
        let meta_dev = block_device_spdk_nvme_v2::BlockDeviceSpdkNvmeComponentV2::new_default();
        meta_dev
            .logger
            .connect(Arc::clone(&log))
            .map_err(|e| PyRuntimeError::new_err(format!("failed to wire logger for metadata device: {e}")))?;
        meta_dev
            .spdk_env
            .connect(Arc::clone(&spdk_iface))
            .map_err(|e| PyRuntimeError::new_err(format!("failed to wire spdk_env for metadata device: {e}")))?;
        let meta_admin: Arc<dyn IBlockDeviceAdmin + Send + Sync> =
            query_interface!(meta_dev, IBlockDeviceAdmin)
                .ok_or_else(|| PyRuntimeError::new_err("failed to query IBlockDeviceAdmin for metadata device"))?;
        let pci = parse_pci_addr(&metadata_pci_addr)
            .map_err(|e| PyRuntimeError::new_err(format!("invalid metadata PCI address '{metadata_pci_addr}': {e}")))?;
        meta_admin.set_pci_address(pci);
        meta_admin
            .initialize()
            .map_err(|e| PyRuntimeError::new_err(format!("metadata block device init failed: {e}")))?;
        let meta_ibd: Arc<dyn IBlockDevice + Send + Sync> =
            query_interface!(meta_dev, IBlockDevice)
                .ok_or_else(|| PyRuntimeError::new_err("failed to query IBlockDevice for metadata device"))?;

        // --- Create extent manager for metadata device ---
        let meta_em = extent_manager_v2::ExtentManagerV2::new_inner();
        let numa_node = meta_ibd.numa_node();
        let dma_alloc: DmaAllocFn = Arc::new(move |size, align, _numa| {
            DmaBuffer::new(size, align, Some(numa_node)).map_err(|e| e.to_string())
        });
        meta_em.set_dma_alloc(dma_alloc);
        meta_em
            .logger
            .connect(Arc::clone(&log) as Arc<dyn ILogger + Send + Sync>)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to wire logger for metadata extent manager: {e}")))?;
        use component_core::binding::bind;
        bind(
            &*meta_dev,
            "IBlockDevice",
            &*meta_em as &dyn component_core::IUnknown,
            "metadata_device",
        )
        .map_err(|e| PyRuntimeError::new_err(format!("failed to bind metadata block device to extent manager: {e}")))?;
        let meta_iem: Arc<dyn IExtentManager + Send + Sync> =
            query_interface!(meta_em, IExtentManager)
                .ok_or_else(|| PyRuntimeError::new_err("failed to query IExtentManager for metadata device"))?;
        let sector_size = meta_ibd.block_size();
        let num_sectors = meta_ibd.num_sectors(1).unwrap_or(0);
        let data_disk_size = num_sectors * sector_size as u64;
        let defaults = FormatParams::default();
        meta_iem
            .format(FormatParams {
                data_disk_size,
                sector_size,
                ..defaults
            })
            .map_err(|e| PyRuntimeError::new_err(format!("metadata extent manager format failed: {e}")))?;

        // --- Create dispatch map, wire extent manager, initialize ---
        let dm_comp =
            dispatch_map::DispatchMapComponentV0::new(dispatch_map::DispatchMapState::default());
        dm_comp
            .logger
            .connect(Arc::clone(&log) as Arc<dyn ILogger + Send + Sync>)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to wire logger for dispatch map: {e}")))?;
        dm_comp
            .extent_manager
            .connect(Arc::clone(&meta_iem))
            .map_err(|e| PyRuntimeError::new_err(format!("failed to wire extent_manager to dispatch map: {e}")))?;
        let dm: Arc<dyn IDispatchMap + Send + Sync> = query_interface!(dm_comp, IDispatchMap)
            .ok_or_else(|| PyRuntimeError::new_err("failed to query IDispatchMap"))?;
        dm.initialize()
            .map_err(|e| PyRuntimeError::new_err(format!("DispatchMap init failed: {e}")))?;

        // --- Create dispatcher ---
        let disp_comp = dispatcher::DispatcherComponentV0::new_default();
        disp_comp
            .logger
            .connect(Arc::clone(&log) as Arc<dyn ILogger + Send + Sync>)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to wire logger for dispatcher: {e}")))?;
        disp_comp
            .dispatch_map
            .connect(Arc::clone(&dm))
            .map_err(|e| PyRuntimeError::new_err(format!("failed to bind dispatch_map: {e}")))?;
        disp_comp
            .gpu_services
            .connect(Arc::clone(&gpu))
            .map_err(|e| PyRuntimeError::new_err(format!("failed to bind gpu_services: {e}")))?;
        disp_comp
            .spdk_env
            .connect(Arc::clone(&spdk_iface))
            .map_err(|e| PyRuntimeError::new_err(format!("failed to bind spdk_env: {e}")))?;

        let dispatcher: Arc<dyn IDispatcher + Send + Sync> =
            query_interface!(disp_comp, IDispatcher)
                .ok_or_else(|| PyRuntimeError::new_err("failed to query IDispatcher"))?;

        dispatcher
            .initialize(DispatcherConfig {
                metadata_pci_addr,
                data_pci_addrs,
                block_device_version: interfaces::BlockDeviceVersion::V2,
                extent_manager_version: interfaces::ExtentManagerVersion::V2,
                max_cache_entries,
                eviction_threshold,
                format_on_init: true,
                ..Default::default()
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Dispatcher init failed: {e}")))?;

        let eviction_watermark =
            (max_cache_entries as f64 * eviction_threshold) as usize;

        Ok(Self {
            dispatcher,
            dispatch_map: dm,
            gpu_services: gpu,
            gpu_block_size,
            max_cache_entries,
            eviction_watermark,
            entry_count: AtomicU64::new(0),
            jobs: Mutex::new(HashMap::new()),
            next_internal_id: AtomicU64::new(0),
            initialized: AtomicBool::new(true),
        })
    }

    fn ensure_init(&self) -> PyResult<()> {
        if !self.initialized.load(Ordering::Acquire) {
            return Err(PyRuntimeError::new_err("engine not initialized"));
        }
        Ok(())
    }

    // ─── Manager-level operations ──────────────────────────────────────

    /// Return count of consecutive keys (from the start) that are cached.
    pub fn batch_check(&self, keys: &[u64]) -> PyResult<u64> {
        self.ensure_init()?;
        let cache_keys = keys::to_cache_keys(keys);
        let mut count: u64 = 0;
        for key in &cache_keys {
            match self.dispatcher.check(*key) {
                Ok(true) => count += 1,
                Ok(false) => break,
                Err(_) => break,
            }
        }
        Ok(count)
    }

    /// Allocate space for new keys, evicting LRU entries if necessary.
    /// Returns (keys_to_store, evicted_keys), or None if eviction cannot
    /// free enough space.
    pub fn prepare_store(&self, keys: &[u64]) -> PyResult<Option<(Vec<u64>, Vec<u64>)>> {
        self.ensure_init()?;
        let cache_keys = keys::to_cache_keys(keys);
        let mut to_store = Vec::new();
        let protected: std::collections::HashSet<CacheKey> =
            cache_keys.iter().copied().collect();

        for (i, key) in cache_keys.iter().enumerate() {
            match self.dispatcher.check(*key) {
                Ok(true) => {}
                Ok(false) | Err(_) => {
                    to_store.push(keys[i]);
                }
            }
        }

        if to_store.is_empty() {
            return Ok(Some((vec![], vec![])));
        }

        let current_count = self.entry_count.load(Ordering::Acquire) as usize;
        let after_store = current_count + to_store.len();
        let evicted = if after_store > self.eviction_watermark {
            let needed = after_store - self.eviction_watermark;
            let candidates = self.dispatch_map.oldest_keys(usize::MAX);
            let mut evicted_keys: Vec<u64> = Vec::new();
            for candidate in candidates {
                if evicted_keys.len() >= needed {
                    break;
                }
                if protected.contains(&candidate) {
                    continue;
                }
                match self.dispatcher.remove(candidate) {
                    Ok(()) => {
                        self.entry_count.fetch_sub(1, Ordering::Release);
                        evicted_keys.push(candidate);
                    }
                    Err(_) => continue,
                }
            }
            if evicted_keys.len() < needed {
                return Ok(None);
            }
            evicted_keys
        } else {
            vec![]
        };

        Ok(Some((to_store, evicted)))
    }

    /// Finalize or abort a store operation.
    pub fn complete_store(&self, keys: &[u64], success: bool) -> PyResult<()> {
        self.ensure_init()?;
        if !success {
            let cache_keys = keys::to_cache_keys(keys);
            for key in &cache_keys {
                if self.dispatcher.remove(*key).is_ok() {
                    self.entry_count.fetch_sub(1, Ordering::Release);
                }
            }
        }
        Ok(())
    }

    /// Update LRU ordering for the given keys.
    pub fn touch(&self, keys: &[u64]) -> PyResult<()> {
        self.ensure_init()?;
        let cache_keys = keys::to_cache_keys(keys);
        for key in &cache_keys {
            let _ = self.dispatcher.touch(*key);
        }
        Ok(())
    }

    /// Pin blocks for reading (protect from eviction) and return their
    /// storage offsets. Assumes all keys are already stored and ready.
    ///
    /// Uses `dispatch_map.lookup()` which atomically increments `read_ref`
    /// and returns the block location. Blocks with `read_ref > 0` cannot
    /// be evicted or removed.
    /// Caller MUST call `complete_load` when DMA is done.
    pub fn prepare_load(&self, keys: &[u64]) -> PyResult<Vec<u64>> {
        self.ensure_init()?;
        let cache_keys = keys::to_cache_keys(keys);
        let mut offsets = Vec::with_capacity(cache_keys.len());

        for (i, key) in cache_keys.iter().enumerate() {
            match self.dispatch_map.lookup(*key) {
                Ok(LookupResult::BlockDevice { offset }) => {
                    offsets.push(offset);
                }
                Ok(LookupResult::MemoryTier { .. }) => {
                    offsets.push(*key);
                }
                Ok(LookupResult::Staging { .. }) => {
                    offsets.push(*key);
                }
                Ok(LookupResult::NotExist) => {
                    // Rollback: release reads we already took
                    for prev_key in &cache_keys[..i] {
                        let _ = self.dispatch_map.release_read(*prev_key);
                    }
                    return Err(PyRuntimeError::new_err(format!(
                        "prepare_load: key {key} not found"
                    )));
                }
                Ok(LookupResult::MismatchSize) => {
                    for prev_key in &cache_keys[..i] {
                        let _ = self.dispatch_map.release_read(*prev_key);
                    }
                    return Err(PyRuntimeError::new_err(format!(
                        "prepare_load: key {key} size mismatch"
                    )));
                }
                Err(e) => {
                    // Rollback: release reads we already took
                    for prev_key in &cache_keys[..i] {
                        let _ = self.dispatch_map.release_read(*prev_key);
                    }
                    return Err(PyRuntimeError::new_err(format!(
                        "prepare_load: lookup failed for key {key}: {e:?}"
                    )));
                }
            }
        }

        Ok(offsets)
    }

    /// Unpin blocks after load DMA completes. Decrements `read_ref` so
    /// blocks become eligible for eviction again.
    pub fn complete_load(&self, keys: &[u64]) -> PyResult<()> {
        self.ensure_init()?;
        let cache_keys = keys::to_cache_keys(keys);

        for key in &cache_keys {
            self.dispatch_map.release_read(*key).map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "complete_load: release_read failed for key {key}: {e:?}"
                ))
            })?;
        }

        Ok(())
    }

    // ─── Handler-level operations ──────────────────────────────────────

    /// Submit async GPU→DRAM→NVMe transfer (store).
    pub fn store_async(&self, job_id: u64, gpu_block_ids: &[u64], keys: &[u64]) -> PyResult<bool> {
        self.ensure_init()?;

        if gpu_block_ids.len() != keys.len() {
            return Err(PyRuntimeError::new_err(
                "gpu_block_ids and keys must have same length",
            ));
        }

        let cache_keys = keys::to_cache_keys(keys);

        let job = Arc::new(TransferJob {
            kind: JobKind::Store,
            keys: cache_keys.clone(),
            gpu_block_ids: gpu_block_ids.to_vec(),
            completed: AtomicBool::new(false),
            success: AtomicBool::new(false),
        });

        {
            let mut jobs = self.jobs.lock().unwrap();
            jobs.insert(job_id, Arc::clone(&job));
        }

        // Execute store: for each block, create an IpcHandle pointing at the
        // GPU memory region and call dispatcher.populate().
        let mut all_ok = true;
        for (i, key) in cache_keys.iter().enumerate() {
            let block_id = gpu_block_ids[i];
            let offset = block_id * self.gpu_block_size;

            // IpcHandle points to GPU memory at the computed offset.
            // The dispatcher will DMA from this address into its staging buffer.
            let handle = IpcHandle {
                address: offset as *mut u8,
                size: self.gpu_block_size as u32,
            };

            if let Err(_e) = self.dispatcher.populate(*key, handle) {
                all_ok = false;
                break;
            }
            self.entry_count.fetch_add(1, Ordering::Release);
        }

        job.completed.store(true, Ordering::Release);
        job.success.store(all_ok, Ordering::Release);

        Ok(all_ok)
    }

    /// Submit async NVMe/DRAM→GPU transfer (load).
    pub fn load_async(&self, job_id: u64, gpu_block_ids: &[u64], keys: &[u64]) -> PyResult<bool> {
        self.ensure_init()?;

        if gpu_block_ids.len() != keys.len() {
            return Err(PyRuntimeError::new_err(
                "gpu_block_ids and keys must have same length",
            ));
        }

        let cache_keys = keys::to_cache_keys(keys);

        let job = Arc::new(TransferJob {
            kind: JobKind::Load,
            keys: cache_keys.clone(),
            gpu_block_ids: gpu_block_ids.to_vec(),
            completed: AtomicBool::new(false),
            success: AtomicBool::new(false),
        });

        {
            let mut jobs = self.jobs.lock().unwrap();
            jobs.insert(job_id, Arc::clone(&job));
        }

        // Execute load: for each block, create an IpcHandle pointing at the
        // destination GPU memory and call dispatcher.lookup().
        let mut all_ok = true;
        for (i, key) in cache_keys.iter().enumerate() {
            let block_id = gpu_block_ids[i];
            let offset = block_id * self.gpu_block_size;

            let handle = IpcHandle {
                address: offset as *mut u8,
                size: self.gpu_block_size as u32,
            };

            if let Err(_e) = self.dispatcher.lookup(*key, handle) {
                all_ok = false;
                break;
            }
        }

        job.completed.store(true, Ordering::Release);
        job.success.store(all_ok, Ordering::Release);

        Ok(all_ok)
    }

    /// Poll for completed transfers. Returns list of (job_id, success).
    pub fn poll_completions(&self) -> PyResult<Vec<(u64, bool)>> {
        self.ensure_init()?;
        let mut completions = Vec::new();
        let mut jobs = self.jobs.lock().unwrap();

        let completed_ids: Vec<u64> = jobs
            .iter()
            .filter(|(_, job)| job.completed.load(Ordering::Acquire))
            .map(|(id, _)| *id)
            .collect();

        for id in completed_ids {
            if let Some(job) = jobs.remove(&id) {
                completions.push((id, job.success.load(Ordering::Acquire)));
            }
        }

        Ok(completions)
    }

    /// Block until a specific job completes.
    pub fn wait_job(&self, job_id: u64) -> PyResult<()> {
        self.ensure_init()?;
        // Jobs complete synchronously in the current implementation,
        // so this is effectively a lookup + remove.
        let jobs = self.jobs.lock().unwrap();
        if let Some(job) = jobs.get(&job_id) {
            if !job.completed.load(Ordering::Acquire) {
                drop(jobs);
                // Spin-wait (will be replaced with condvar when async I/O lands)
                loop {
                    let jobs = self.jobs.lock().unwrap();
                    if let Some(job) = jobs.get(&job_id) {
                        if job.completed.load(Ordering::Acquire) {
                            break;
                        }
                    } else {
                        break;
                    }
                    drop(jobs);
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
            }
        }
        Ok(())
    }

    /// Store bytes from a host buffer directly (no GPU DMA). For testing only.
    /// Uses dispatcher.prepare_store()+commit_store() to write directly into
    /// the DMA buffer and flush to NVMe without going through CUDA.
    pub fn store_host_bytes(&self, key: u64, data: &[u8]) -> PyResult<()> {
        self.ensure_init()?;
        let dma_buf = self.dispatcher
            .prepare_store(key, data.len() as u32)
            .map_err(|e| PyRuntimeError::new_err(format!("store_host_bytes prepare failed: {e}")))?;

        // Copy data into the DMA buffer directly.
        // SAFETY: dma_buf is a valid DMA allocation covering at least data.len() bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), dma_buf.as_ptr() as *mut u8, data.len());
        }

        self.dispatcher
            .commit_store(key)
            .map_err(|e| PyRuntimeError::new_err(format!("store_host_bytes commit failed: {e}")))?;

        self.entry_count.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Read bytes from the dispatch map's staging buffer for a key (no GPU DMA).
    /// For testing only — verifies data was written correctly before background
    /// NVMe migration moves it off the staging buffer.
    pub fn load_host_bytes(&self, key: u64, size: usize) -> PyResult<Vec<u8>> {
        self.ensure_init()?;
        // Read directly from the dispatch map staging buffer, bypassing GPU DMA.
        let result = self.dispatch_map
            .lookup(key)
            .map_err(|e| PyRuntimeError::new_err(format!("load_host_bytes lookup failed: {e}")))?;

        use interfaces::LookupResult;
        match result {
            LookupResult::Staging { buffer } => {
                let copy_len = size.min(buffer.len());
                let mut out = vec![0u8; size];
                // SAFETY: buffer is a valid DMA allocation; out is a valid heap allocation.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buffer.as_ptr() as *const u8,
                        out.as_mut_ptr(),
                        copy_len,
                    );
                }
                let _ = self.dispatch_map.release_read(key);
                Ok(out)
            }
            LookupResult::MemoryTier { pointer, size: entry_size } => {
                let copy_len = size.min(entry_size as usize);
                let mut out = vec![0u8; size];
                // SAFETY: pointer is a valid memory-tier slot for entry_size bytes.
                unsafe {
                    std::ptr::copy_nonoverlapping(pointer, out.as_mut_ptr(), copy_len);
                }
                let _ = self.dispatch_map.release_read(key);
                Ok(out)
            }
            LookupResult::BlockDevice { .. } => {
                let _ = self.dispatch_map.release_read(key);
                Err(PyRuntimeError::new_err(
                    "key already migrated to NVMe — use load_async for block device reads",
                ))
            }
            LookupResult::NotExist => Err(PyRuntimeError::new_err(
                format!("key {key} not found in dispatch map"),
            )),
            LookupResult::MismatchSize => {
                let _ = self.dispatch_map.release_read(key);
                Err(PyRuntimeError::new_err("size mismatch on lookup"))
            }
        }
    }

    /// Shut down the engine, releasing all resources.
    pub fn shutdown(&self) -> PyResult<()> {
        if !self.initialized.swap(false, Ordering::AcqRel) {
            return Ok(());
        }

        self.dispatcher
            .shutdown()
            .map_err(|e| PyRuntimeError::new_err(format!("dispatcher shutdown failed: {e}")))?;

        self.gpu_services
            .shutdown()
            .map_err(|e| PyRuntimeError::new_err(format!("GPU shutdown failed: {e}")))?;

        Ok(())
    }
}


#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    use interfaces::{
        DispatchMapError, DispatcherError, DmaBuffer, GpuDeviceInfo, GpuDmaBuffer,
        GpuIpcHandle, LookupResult,
    };

    #[derive(Default)]
    struct MockDispatcherState {
        check_results: HashMap<CacheKey, Result<bool, DispatcherError>>,
        remove_results: HashMap<CacheKey, Result<(), DispatcherError>>,
        touch_results: HashMap<CacheKey, Result<(), DispatcherError>>,
        populate_results: HashMap<CacheKey, Result<(), DispatcherError>>,
        lookup_results: HashMap<CacheKey, Result<(), DispatcherError>>,
        populate_calls: Vec<(CacheKey, usize, u32)>,
        lookup_calls: Vec<(CacheKey, usize, u32)>,
        shutdown_calls: u64,
    }

    pub(crate) struct MockDispatcher {
        state: Mutex<MockDispatcherState>,
    }

    impl MockDispatcher {
        pub(crate) fn new() -> Self {
            Self {
                state: Mutex::new(MockDispatcherState::default()),
            }
        }

        pub(crate) fn set_check_result(&self, key: CacheKey, result: Result<bool, DispatcherError>) {
            self.state.lock().unwrap().check_results.insert(key, result);
        }

        pub(crate) fn set_remove_result(&self, key: CacheKey, result: Result<(), DispatcherError>) {
            self.state.lock().unwrap().remove_results.insert(key, result);
        }

        pub(crate) fn set_touch_result(&self, key: CacheKey, result: Result<(), DispatcherError>) {
            self.state.lock().unwrap().touch_results.insert(key, result);
        }

        pub(crate) fn set_populate_result(&self, key: CacheKey, result: Result<(), DispatcherError>) {
            self.state.lock().unwrap().populate_results.insert(key, result);
        }

        pub(crate) fn set_lookup_result(&self, key: CacheKey, result: Result<(), DispatcherError>) {
            self.state.lock().unwrap().lookup_results.insert(key, result);
        }

        pub(crate) fn populate_calls(&self) -> Vec<(CacheKey, usize, u32)> {
            self.state.lock().unwrap().populate_calls.clone()
        }

        pub(crate) fn lookup_calls(&self) -> Vec<(CacheKey, usize, u32)> {
            self.state.lock().unwrap().lookup_calls.clone()
        }

        pub(crate) fn shutdown_calls(&self) -> u64 {
            self.state.lock().unwrap().shutdown_calls
        }
    }

    impl IDispatcher for MockDispatcher {
        fn initialize(&self, _config: DispatcherConfig) -> Result<(), DispatcherError> {
            Ok(())
        }

        fn shutdown(&self) -> Result<(), DispatcherError> {
            self.state.lock().unwrap().shutdown_calls += 1;
            Ok(())
        }

        fn lookup(&self, key: CacheKey, ipc_handle: IpcHandle) -> Result<(), DispatcherError> {
            let mut state = self.state.lock().unwrap();
            state.lookup_calls.push((key, ipc_handle.address as usize, ipc_handle.size));
            state.lookup_results.get(&key).cloned().unwrap_or(Ok(()))
        }

        fn check(&self, key: CacheKey) -> Result<bool, DispatcherError> {
            self.state.lock().unwrap().check_results.get(&key).cloned().unwrap_or(Ok(false))
        }

        fn remove(&self, key: CacheKey) -> Result<(), DispatcherError> {
            self.state.lock().unwrap().remove_results.get(&key).cloned().unwrap_or(Ok(()))
        }

        fn populate(&self, key: CacheKey, ipc_handle: IpcHandle) -> Result<(), DispatcherError> {
            let mut state = self.state.lock().unwrap();
            state.populate_calls.push((key, ipc_handle.address as usize, ipc_handle.size));
            state.populate_results.get(&key).cloned().unwrap_or(Ok(()))
        }

        fn prepare_store(&self, _key: CacheKey, _size: u32) -> Result<Arc<DmaBuffer>, DispatcherError> {
            Err(DispatcherError::InvalidParameter("unused in engine tests".into()))
        }

        fn commit_store(&self, _key: CacheKey) -> Result<(), DispatcherError> {
            Ok(())
        }

        fn cancel_store(&self, _key: CacheKey) -> Result<(), DispatcherError> {
            Ok(())
        }

        fn touch(&self, key: CacheKey) -> Result<(), DispatcherError> {
            self.state.lock().unwrap().touch_results.get(&key).cloned().unwrap_or(Ok(()))
        }
    }

    #[derive(Clone, Copy)]
    pub(crate) enum MockLookupResult {
        NotExist,
        Mismatch,
        BlockDevice(u64),
    }

    #[derive(Default)]
    struct MockDispatchMapState {
        lookup_results: HashMap<CacheKey, MockLookupResult>,
        oldest_keys: Vec<CacheKey>,
        release_read_calls: Vec<CacheKey>,
    }

    pub(crate) struct MockDispatchMap {
        state: Mutex<MockDispatchMapState>,
    }

    impl MockDispatchMap {
        pub(crate) fn new() -> Self {
            Self {
                state: Mutex::new(MockDispatchMapState::default()),
            }
        }

        pub(crate) fn set_lookup_result(&self, key: CacheKey, result: MockLookupResult) {
            self.state.lock().unwrap().lookup_results.insert(key, result);
        }

        pub(crate) fn set_oldest_keys(&self, keys: Vec<CacheKey>) {
            self.state.lock().unwrap().oldest_keys = keys;
        }

        pub(crate) fn release_read_calls(&self) -> Vec<CacheKey> {
            self.state.lock().unwrap().release_read_calls.clone()
        }
    }

    impl IDispatchMap for MockDispatchMap {
        fn set_dma_alloc(&self, _alloc: DmaAllocFn) {}

        fn initialize(&self) -> Result<(), DispatchMapError> {
            Ok(())
        }

        fn create_staging(&self, _key: CacheKey, _size: u32) -> Result<Arc<DmaBuffer>, DispatchMapError> {
            Err(DispatchMapError::NotInitialized("unused in engine tests".into()))
        }

        fn lookup(&self, key: CacheKey) -> Result<LookupResult, DispatchMapError> {
            match self.state.lock().unwrap().lookup_results.get(&key).copied().unwrap_or(MockLookupResult::NotExist) {
                MockLookupResult::NotExist => Ok(LookupResult::NotExist),
                MockLookupResult::Mismatch => Ok(LookupResult::MismatchSize),
                MockLookupResult::BlockDevice(offset) => Ok(LookupResult::BlockDevice { offset }),
            }
        }

        fn convert_to_storage(&self, _key: CacheKey, _offset: u64) -> Result<(), DispatchMapError> {
            Ok(())
        }

        fn take_read(&self, _key: CacheKey) -> Result<(), DispatchMapError> {
            Ok(())
        }

        fn take_write(&self, _key: CacheKey) -> Result<(), DispatchMapError> {
            Ok(())
        }

        fn release_read(&self, key: CacheKey) -> Result<(), DispatchMapError> {
            self.state.lock().unwrap().release_read_calls.push(key);
            Ok(())
        }

        fn release_write(&self, _key: CacheKey) -> Result<(), DispatchMapError> {
            Ok(())
        }

        fn downgrade_reference(&self, _key: CacheKey) -> Result<(), DispatchMapError> {
            Ok(())
        }

        fn remove(&self, _key: CacheKey) -> Result<(), DispatchMapError> {
            Ok(())
        }

        fn touch(&self, _key: CacheKey) -> Result<(), DispatchMapError> {
            Ok(())
        }

        fn oldest_keys(&self, n: usize) -> Vec<CacheKey> {
            self.state.lock().unwrap().oldest_keys.iter().copied().take(n).collect()
        }

        fn create_memory_tier_entry(&self, _key: CacheKey, _pointer: *mut u8, _size: u32) -> Result<(), DispatchMapError> {
            Err(DispatchMapError::NotInitialized("unused in engine tests".into()))
        }

        fn convert_memory_tier_to_block(&self, _key: CacheKey) -> Result<(), DispatchMapError> {
            Err(DispatchMapError::NotInitialized("unused in engine tests".into()))
        }
    }

    pub(crate) struct MockGpuServices {
        shutdown_calls: AtomicU64,
    }

    impl MockGpuServices {
        pub(crate) fn new() -> Self {
            Self {
                shutdown_calls: AtomicU64::new(0),
            }
        }

        pub(crate) fn shutdown_calls(&self) -> u64 {
            self.shutdown_calls.load(Ordering::Acquire)
        }
    }

    impl IGpuServices for MockGpuServices {
        fn initialize(&self) -> Result<(), String> {
            Ok(())
        }

        fn shutdown(&self) -> Result<(), String> {
            self.shutdown_calls.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }

        fn get_devices(&self) -> Result<Vec<GpuDeviceInfo>, String> {
            Ok(vec![])
        }

        fn deserialize_ipc_handle(&self, _base64_payload: &str) -> Result<GpuIpcHandle, String> {
            Err("unused in engine tests".into())
        }

        fn verify_memory(&self, _handle: &GpuIpcHandle) -> Result<(), String> {
            Ok(())
        }

        fn pin_memory(&self, _handle: &GpuIpcHandle) -> Result<(), String> {
            Ok(())
        }

        fn unpin_memory(&self, _handle: &GpuIpcHandle) -> Result<(), String> {
            Ok(())
        }

        fn create_dma_buffer(&self, _handle: GpuIpcHandle) -> Result<GpuDmaBuffer, String> {
            Err("unused in engine tests".into())
        }

        fn dma_copy_to_host(&self, _src: *const std::ffi::c_void, _dst: &DmaBuffer, _size: usize) -> Result<(), String> {
            Ok(())
        }

        fn dma_copy_to_device(&self, _src: &DmaBuffer, _dst: *mut std::ffi::c_void, _size: usize) -> Result<(), String> {
            Ok(())
        }

        fn prepare_memory_for_spdk(&self, _base64_payload: &str, _device_index: Option<u32>) -> Result<DmaBuffer, String> {
            Err("unused in engine tests".into())
        }
    }

    pub(crate) fn build_engine(
        dispatcher: Arc<dyn IDispatcher + Send + Sync>,
        dispatch_map: Arc<dyn IDispatchMap + Send + Sync>,
        gpu_services: Arc<dyn IGpuServices + Send + Sync>,
        gpu_block_size: u64,
        max_cache_entries: usize,
        eviction_watermark: usize,
        entry_count: u64,
        initialized: bool,
    ) -> EngineInner {
        EngineInner {
            dispatcher,
            dispatch_map,
            gpu_services,
            gpu_block_size,
            max_cache_entries,
            eviction_watermark,
            entry_count: AtomicU64::new(entry_count),
            jobs: Mutex::new(HashMap::new()),
            next_internal_id: AtomicU64::new(0),
            initialized: AtomicBool::new(initialized),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    use super::test_support::{
        build_engine, MockDispatchMap, MockDispatcher, MockGpuServices, MockLookupResult,
    };
    use interfaces::DispatcherError;

    #[test]
    fn parse_pci_addr_parses_components() {
        let addr = parse_pci_addr("0000:2a:1f.3").expect("valid PCI address");
        assert_eq!(addr.domain, 0);
        assert_eq!(addr.bus, 0x2a);
        assert_eq!(addr.dev, 0x1f);
        assert_eq!(addr.func, 3);
    }

    #[test]
    fn parse_pci_addr_rejects_invalid_input() {
        let err = parse_pci_addr("0000:2a:zz.3").expect_err("invalid device should fail");
        assert!(err.contains("invalid dev"));
    }

    #[test]
    fn batch_check_counts_until_first_miss() {
        let dispatcher = Arc::new(MockDispatcher::new());
        dispatcher.set_check_result(1, Ok(true));
        dispatcher.set_check_result(2, Ok(true));
        let engine = build_engine(
            dispatcher.clone(),
            Arc::new(MockDispatchMap::new()),
            Arc::new(MockGpuServices::new()),
            4096,
            16,
            16,
            0,
            true,
        );

        assert_eq!(engine.batch_check(&[1, 2, 3, 4]).unwrap(), 2);
    }

    #[test]
    fn prepare_store_evicts_oldest_unprotected_key() {
        let dispatcher = Arc::new(MockDispatcher::new());
        dispatcher.set_check_result(1, Ok(true));
        dispatcher.set_check_result(2, Ok(false));
        let dispatch_map = Arc::new(MockDispatchMap::new());
        dispatch_map.set_oldest_keys(vec![99, 2]);
        let engine = build_engine(
            dispatcher.clone(),
            dispatch_map,
            Arc::new(MockGpuServices::new()),
            4096,
            4,
            1,
            1,
            true,
        );

        let result = engine.prepare_store(&[1, 2]).unwrap();
        assert_eq!(result, Some((vec![2], vec![99])));
        assert_eq!(engine.entry_count.load(Ordering::Acquire), 0);
    }

    #[test]
    fn prepare_load_releases_prior_reads_on_error() {
        let dispatch_map = Arc::new(MockDispatchMap::new());
        dispatch_map.set_lookup_result(10, MockLookupResult::BlockDevice(4096));
        dispatch_map.set_lookup_result(11, MockLookupResult::NotExist);
        let engine = build_engine(
            Arc::new(MockDispatcher::new()),
            dispatch_map.clone(),
            Arc::new(MockGpuServices::new()),
            4096,
            8,
            8,
            0,
            true,
        );

        let err = engine.prepare_load(&[10, 11]).expect_err("missing key should fail");
        assert!(err.to_string().contains("key 11 not found"));
        assert_eq!(dispatch_map.release_read_calls(), vec![10]);
    }

    #[test]
    fn store_async_records_offsets_and_completions() {
        let dispatcher = Arc::new(MockDispatcher::new());
        let engine = build_engine(
            dispatcher.clone(),
            Arc::new(MockDispatchMap::new()),
            Arc::new(MockGpuServices::new()),
            4096,
            16,
            16,
            0,
            true,
        );

        let ok = engine.store_async(7, &[3, 5], &[101, 102]).unwrap();
        assert!(ok);
        assert_eq!(
            dispatcher.populate_calls(),
            vec![(101, 3 * 4096, 4096), (102, 5 * 4096, 4096)]
        );
        assert_eq!(engine.entry_count.load(Ordering::Acquire), 2);
        assert_eq!(engine.poll_completions().unwrap(), vec![(7, true)]);
        assert!(engine.poll_completions().unwrap().is_empty());
    }

    #[test]
    fn load_async_reports_dispatch_failures() {
        let dispatcher = Arc::new(MockDispatcher::new());
        dispatcher.set_lookup_result(202, Err(DispatcherError::IoError("boom".into())));
        let engine = build_engine(
            dispatcher.clone(),
            Arc::new(MockDispatchMap::new()),
            Arc::new(MockGpuServices::new()),
            4096,
            16,
            16,
            0,
            true,
        );

        let ok = engine.load_async(9, &[1, 2], &[201, 202]).unwrap();
        assert!(!ok);
        assert_eq!(
            dispatcher.lookup_calls(),
            vec![(201, 4096, 4096), (202, 8192, 4096)]
        );
        assert_eq!(engine.poll_completions().unwrap(), vec![(9, false)]);
    }

    #[test]
    fn shutdown_only_runs_once() {
        let dispatcher = Arc::new(MockDispatcher::new());
        let gpu = Arc::new(MockGpuServices::new());
        let engine = build_engine(
            dispatcher.clone(),
            Arc::new(MockDispatchMap::new()),
            gpu.clone(),
            4096,
            16,
            16,
            0,
            true,
        );

        engine.shutdown().unwrap();
        engine.shutdown().unwrap();

        assert_eq!(dispatcher.shutdown_calls(), 1);
        assert_eq!(gpu.shutdown_calls(), 1);
    }
}
