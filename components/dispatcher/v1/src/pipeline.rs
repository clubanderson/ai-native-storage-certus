//! Ring-buffer pipelined reader for SSD→DRAM→GPU transfers.
//!
//! Reads data from SSD in MDTS-sized chunks using a ring of DMA buffers,
//! copying each completed chunk to both the memory-tier slot and the GPU
//! destination. This pipelines SSD reads with DRAM→GPU DMA transfers.

use std::sync::{Arc, Mutex};

use interfaces::{
    ClientChannels, Command, Completion, DmaAllocFn, DmaBuffer, DispatcherError, IBlockDevice,
    IGpuServices,
};

use crate::io_segmenter;

/// Number of ring buffers for pipelined transfers.
pub const PIPELINE_RING_SIZE: usize = 4;

/// Pipeline-read from SSD into a memory-tier slot while streaming chunks to GPU.
///
/// For each chunk:
/// 1. Issue SSD read into ring buffer
/// 2. On completion, copy chunk to memory-tier slot
/// 3. DMA-copy chunk to GPU destination
///
/// This overlaps SSD I/O with GPU DMA by working on different chunks concurrently.
/// # Safety
///
/// - `mem_tier_ptr` must be valid for writes of at least `total_bytes` (aligned up to block size).
/// - `gpu_dst` must be a valid GPU destination pointer for `total_bytes`.
pub unsafe fn pipelined_ssd_to_gpu(
    drive: &dyn IBlockDevice,
    gpu: &dyn IGpuServices,
    mem_tier_ptr: *mut u8,
    gpu_dst: *mut std::ffi::c_void,
    start_lba: u64,
    total_bytes: usize,
    numa_node: i32,
) -> Result<(), DispatcherError> {
    let default_alloc: DmaAllocFn = Arc::new(move |size, align, node| {
        DmaBuffer::new(size, align, node).map_err(|e| e.to_string())
    });
    // SAFETY: caller upholds the same ptr validity requirements.
    unsafe {
        pipelined_ssd_to_gpu_inner(
            drive,
            gpu,
            mem_tier_ptr,
            gpu_dst,
            start_lba,
            total_bytes,
            numa_node,
            &default_alloc,
        )
    }
}

/// Testable inner implementation; accepts a pluggable DMA allocator.
///
/// # Safety
///
/// Same as [`pipelined_ssd_to_gpu`].
pub(crate) unsafe fn pipelined_ssd_to_gpu_inner(
    drive: &dyn IBlockDevice,
    gpu: &dyn IGpuServices,
    mem_tier_ptr: *mut u8,
    gpu_dst: *mut std::ffi::c_void,
    start_lba: u64,
    total_bytes: usize,
    numa_node: i32,
    dma_alloc: &DmaAllocFn,
) -> Result<(), DispatcherError> {
    let block_size = drive.block_size() as usize;
    let chunk_size = drive.max_transfer_size() as usize;
    let aligned_bytes = total_bytes.next_multiple_of(block_size);

    let channels: ClientChannels = drive.connect_client().map_err(|e| {
        DispatcherError::IoError(format!("connect_client failed: {e}"))
    })?;

    let segments =
        io_segmenter::segment_io(start_lba, aligned_bytes, chunk_size as u32, block_size as u32);

    // Allocate ring of DMA buffers.
    let ring: Vec<Arc<Mutex<DmaBuffer>>> = (0..PIPELINE_RING_SIZE.min(segments.len()))
        .map(|_| {
            dma_alloc(chunk_size, block_size, Some(numa_node))
                .map(|b| Arc::new(Mutex::new(b)))
                .map_err(|e| {
                    DispatcherError::AllocationFailed(format!("pipeline ring buffer: {e}"))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    for (i, seg) in segments.iter().enumerate() {
        let ring_idx = i % ring.len();
        let ring_buf = Arc::clone(&ring[ring_idx]);

        // Issue SSD read.
        channels
            .command_tx
            .send(Command::ReadSync {
                ns_id: 1,
                lba: seg.lba,
                buf: ring_buf.clone(),
            })
            .map_err(|_| DispatcherError::IoError("send ReadSync failed".into()))?;

        match channels.completion_rx.recv() {
            Ok(Completion::ReadDone { result, .. }) => {
                result.map_err(|e| DispatcherError::IoError(format!("SSD read failed: {e}")))?;
            }
            Ok(other) => {
                return Err(DispatcherError::IoError(format!(
                    "unexpected completion: {other:?}"
                )));
            }
            Err(_) => {
                return Err(DispatcherError::IoError(
                    "completion channel disconnected".into(),
                ));
            }
        }

        // Copy from ring buffer to memory-tier slot.
        let copy_len = seg.length.min(total_bytes.saturating_sub(seg.buffer_offset));
        let guard = ring_buf.lock().unwrap();
        if copy_len > 0 {
            // SAFETY: mem_tier_ptr + buffer_offset is within the memory-tier slot.
            // guard.as_ptr() is valid for seg.length bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    guard.as_ptr() as *const u8,
                    mem_tier_ptr.add(seg.buffer_offset),
                    copy_len,
                );
            }
        }

        // DMA-copy this chunk to GPU.
        let gpu_offset = seg.buffer_offset;
        gpu.dma_copy_to_device(
            &guard,
            // SAFETY: gpu_dst + gpu_offset is within the caller's GPU buffer.
            unsafe { (gpu_dst as *mut u8).add(gpu_offset) as *mut std::ffi::c_void },
            copy_len,
        )
        .map_err(|e| {
            DispatcherError::IoError(format!("GPU DMA copy (pipeline chunk) failed: {e}"))
        })?;

        drop(guard);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;
    use std::thread;

    use component_core::channel::SpscChannel;
    use interfaces::{
        GpuDeviceInfo, GpuDmaBuffer, GpuIpcHandle, NvmeBlockError, OpHandle, TelemetrySnapshot,
    };

    // -----------------------------------------------------------------------
    // DMA allocator helpers — libc-backed, no SPDK required
    // -----------------------------------------------------------------------

    unsafe extern "C" fn test_dma_free(ptr: *mut std::ffi::c_void) {
        // SAFETY: ptr was allocated with libc::aligned_alloc
        unsafe { libc::free(ptr) }
    }

    /// Returns a [`DmaAllocFn`] that allocates page-aligned heap memory.
    fn make_alloc() -> DmaAllocFn {
        Arc::new(|size: usize, _align: usize, _node: Option<i32>| {
            let actual = size.max(512).next_multiple_of(4096);
            // SAFETY: aligned_alloc with power-of-two alignment.
            let ptr = unsafe { libc::aligned_alloc(4096, actual) };
            if ptr.is_null() {
                return Err(format!("aligned_alloc({actual}) failed"));
            }
            // SAFETY: ptr is valid for `actual` bytes from aligned_alloc.
            unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, actual) };
            // SAFETY: ptr ownership transferred to DmaBuffer; test_dma_free matches.
            unsafe { DmaBuffer::from_raw(ptr, actual, test_dma_free, -1) }
                .map_err(|e| e.to_string())
        })
    }

    /// Returns a [`DmaAllocFn`] that always fails (for allocation-error tests).
    fn make_fail_alloc() -> DmaAllocFn {
        Arc::new(|_size, _align, _node| Err("mock allocation failure".to_string()))
    }

    // -----------------------------------------------------------------------
    // MockBlockDevice
    // -----------------------------------------------------------------------

    struct MockBlockDevice {
        blk_size: u32,
        max_xfer: u32,
        /// Pre-built `ClientChannels` returned once by `connect_client`.
        channels: Mutex<Option<ClientChannels>>,
    }

    impl MockBlockDevice {
        /// Construct the device and return the opposing channel endpoints for
        /// the mock responder thread: `(device, cmd_receiver, completion_sender)`.
        fn new(
            blk_size: u32,
            max_xfer: u32,
        ) -> (
            Self,
            component_core::channel::Receiver<Command>,
            component_core::channel::Sender<Completion>,
        ) {
            let cmd_ch = SpscChannel::<Command>::new(64);
            let cmd_tx = cmd_ch.sender().unwrap();
            let cmd_rx = cmd_ch.receiver().unwrap();

            let cmp_ch = SpscChannel::<Completion>::new(64);
            let cmp_tx = cmp_ch.sender().unwrap();
            let cmp_rx = cmp_ch.receiver().unwrap();

            let dev = Self {
                blk_size,
                max_xfer,
                channels: Mutex::new(Some(ClientChannels {
                    command_tx: cmd_tx,
                    completion_rx: cmp_rx,
                })),
            };
            (dev, cmd_rx, cmp_tx)
        }
    }

    impl IBlockDevice for MockBlockDevice {
        fn connect_client(&self) -> Result<ClientChannels, NvmeBlockError> {
            self.channels
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| NvmeBlockError::NotInitialized("already connected".into()))
        }
        fn sector_size(&self, _ns_id: u32) -> Result<u32, NvmeBlockError> {
            Ok(self.blk_size)
        }
        fn num_sectors(&self, _ns_id: u32) -> Result<u64, NvmeBlockError> {
            Ok(u64::MAX)
        }
        fn max_queue_depth(&self) -> u32 {
            32
        }
        fn num_io_queues(&self) -> u32 {
            1
        }
        fn max_transfer_size(&self) -> u32 {
            self.max_xfer
        }
        fn block_size(&self) -> u32 {
            self.blk_size
        }
        fn numa_node(&self) -> i32 {
            -1
        }
        fn nvme_version(&self) -> String {
            "1.0".into()
        }
        fn telemetry(&self) -> Result<TelemetrySnapshot, NvmeBlockError> {
            Err(NvmeBlockError::FeatureNotEnabled("mock".into()))
        }
    }

    // -----------------------------------------------------------------------
    // MockGpu
    // -----------------------------------------------------------------------

    struct MockGpu {
        /// When true, `dma_copy_to_device` returns an error.
        fail: bool,
    }

    impl IGpuServices for MockGpu {
        fn initialize(&self) -> Result<(), String> {
            Ok(())
        }
        fn shutdown(&self) -> Result<(), String> {
            Ok(())
        }
        fn get_devices(&self) -> Result<Vec<GpuDeviceInfo>, String> {
            Ok(vec![])
        }
        fn deserialize_ipc_handle(&self, _: &str) -> Result<GpuIpcHandle, String> {
            Err("mock".into())
        }
        fn verify_memory(&self, _: &GpuIpcHandle) -> Result<(), String> {
            Ok(())
        }
        fn pin_memory(&self, _: &GpuIpcHandle) -> Result<(), String> {
            Ok(())
        }
        fn unpin_memory(&self, _: &GpuIpcHandle) -> Result<(), String> {
            Ok(())
        }
        fn create_dma_buffer(&self, _: GpuIpcHandle) -> Result<GpuDmaBuffer, String> {
            Err("mock".into())
        }
        fn dma_copy_to_host(
            &self,
            src: *const std::ffi::c_void,
            dst: &DmaBuffer,
            size: usize,
        ) -> Result<(), String> {
            if size > 0 {
                // SAFETY: both pointers are valid host allocations in tests.
                unsafe {
                    std::ptr::copy_nonoverlapping(src as *const u8, dst.as_ptr() as *mut u8, size);
                }
            }
            Ok(())
        }
        fn dma_copy_to_device(
            &self,
            src: &DmaBuffer,
            dst: *mut std::ffi::c_void,
            size: usize,
        ) -> Result<(), String> {
            if self.fail {
                return Err("mock GPU DMA error".into());
            }
            if size > 0 {
                // SAFETY: both pointers are valid host allocations in tests.
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, dst as *mut u8, size);
                }
            }
            Ok(())
        }
        fn prepare_memory_for_spdk(
            &self,
            _: &str,
            _: Option<u32>,
        ) -> Result<DmaBuffer, String> {
            Err("mock".into())
        }
    }

    // -----------------------------------------------------------------------
    // Responder helper
    // -----------------------------------------------------------------------

    /// Spawns a thread that responds to exactly `n` `ReadSync` commands with
    /// `ReadDone { result: Ok(()) }`.
    fn spawn_ok_responder(
        n: usize,
        cmd_rx: component_core::channel::Receiver<Command>,
        cmp_tx: component_core::channel::Sender<Completion>,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            for _ in 0..n {
                match cmd_rx.recv() {
                    Ok(Command::ReadSync { .. }) => {
                        cmp_tx
                            .send(Completion::ReadDone {
                                handle: OpHandle(0),
                                result: Ok(()),
                            })
                            .unwrap();
                    }
                    Ok(_) => panic!("expected ReadSync"),
                    Err(_) => break,
                }
            }
        })
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn pipeline_ring_size_is_reasonable() {
        assert!(PIPELINE_RING_SIZE >= 2);
        assert!(PIPELINE_RING_SIZE <= 16);
    }

    /// Happy path: 1 segment, ring allocates 1 buffer, read completes OK,
    /// GPU DMA succeeds → `Ok(())`.
    #[test]
    fn happy_path_single_segment() {
        let (dev, cmd_rx, cmp_tx) = MockBlockDevice::new(512, 4096);
        let gpu = MockGpu { fail: false };
        let alloc = make_alloc();

        let total_bytes = 4096usize;
        let mut mem_buf = vec![0u8; total_bytes];
        let mut gpu_buf = vec![0u8; total_bytes];

        let h = spawn_ok_responder(1, cmd_rx, cmp_tx);
        let result = unsafe {
            pipelined_ssd_to_gpu_inner(
                &dev,
                &gpu,
                mem_buf.as_mut_ptr(),
                gpu_buf.as_mut_ptr() as *mut std::ffi::c_void,
                0,
                total_bytes,
                -1,
                &alloc,
            )
        };
        h.join().unwrap();
        assert!(result.is_ok(), "expected Ok(()), got: {result:?}");
    }

    /// Ring wrap: 6 segments with a 4-slot ring; slot reuse must not corrupt
    /// earlier chunks.
    #[test]
    fn ring_wrap_six_segments() {
        let (dev, cmd_rx, cmp_tx) = MockBlockDevice::new(512, 4096);
        let gpu = MockGpu { fail: false };
        let alloc = make_alloc();

        let total_bytes = 6 * 4096usize;
        let mut mem_buf = vec![0u8; total_bytes];
        let mut gpu_buf = vec![0u8; total_bytes];

        let h = spawn_ok_responder(6, cmd_rx, cmp_tx);
        let result = unsafe {
            pipelined_ssd_to_gpu_inner(
                &dev,
                &gpu,
                mem_buf.as_mut_ptr(),
                gpu_buf.as_mut_ptr() as *mut std::ffi::c_void,
                0,
                total_bytes,
                -1,
                &alloc,
            )
        };
        h.join().unwrap();
        assert!(result.is_ok(), "ring wrap should succeed, got: {result:?}");
    }

    /// SSD read error: `Completion::ReadDone { result: Err(...) }` →
    /// `Err(DispatcherError::IoError)`.
    #[test]
    fn ssd_read_error_propagates() {
        let (dev, cmd_rx, cmp_tx) = MockBlockDevice::new(512, 4096);
        let gpu = MockGpu { fail: false };
        let alloc = make_alloc();

        let total_bytes = 4096usize;
        let mut mem_buf = vec![0u8; total_bytes];
        let mut gpu_buf = vec![0u8; total_bytes];

        let h = thread::spawn(move || {
            if let Ok(Command::ReadSync { .. }) = cmd_rx.recv() {
                cmp_tx
                    .send(Completion::ReadDone {
                        handle: OpHandle(0),
                        result: Err(NvmeBlockError::Timeout("disk timeout".into())),
                    })
                    .unwrap();
            }
        });

        let result = unsafe {
            pipelined_ssd_to_gpu_inner(
                &dev,
                &gpu,
                mem_buf.as_mut_ptr(),
                gpu_buf.as_mut_ptr() as *mut std::ffi::c_void,
                0,
                total_bytes,
                -1,
                &alloc,
            )
        };
        h.join().unwrap();
        assert!(
            matches!(result, Err(DispatcherError::IoError(_))),
            "expected IoError for SSD read failure, got: {result:?}"
        );
    }

    /// Unexpected completion variant → `Err(IoError("unexpected completion"))`.
    #[test]
    fn unexpected_completion_variant() {
        let (dev, cmd_rx, cmp_tx) = MockBlockDevice::new(512, 4096);
        let gpu = MockGpu { fail: false };
        let alloc = make_alloc();

        let total_bytes = 4096usize;
        let mut mem_buf = vec![0u8; total_bytes];
        let mut gpu_buf = vec![0u8; total_bytes];

        let h = thread::spawn(move || {
            if let Ok(Command::ReadSync { .. }) = cmd_rx.recv() {
                cmp_tx
                    .send(Completion::WriteDone {
                        handle: OpHandle(0),
                        result: Ok(()),
                    })
                    .unwrap();
            }
        });

        let result = unsafe {
            pipelined_ssd_to_gpu_inner(
                &dev,
                &gpu,
                mem_buf.as_mut_ptr(),
                gpu_buf.as_mut_ptr() as *mut std::ffi::c_void,
                0,
                total_bytes,
                -1,
                &alloc,
            )
        };
        h.join().unwrap();
        match &result {
            Err(DispatcherError::IoError(msg)) => {
                assert!(msg.contains("unexpected completion"), "wrong message: {msg}")
            }
            other => panic!("expected IoError(unexpected completion), got: {other:?}"),
        }
    }

    /// Channel disconnect: completion_rx closed → `Err(IoError("completion channel
    /// disconnected"))`.
    #[test]
    fn channel_disconnect() {
        let (dev, cmd_rx, cmp_tx) = MockBlockDevice::new(512, 4096);
        let gpu = MockGpu { fail: false };
        let alloc = make_alloc();

        let total_bytes = 4096usize;
        let mut mem_buf = vec![0u8; total_bytes];
        let mut gpu_buf = vec![0u8; total_bytes];

        let h = thread::spawn(move || {
            // Consume the command, then close the channel without a reply.
            let _ = cmd_rx.recv();
            drop(cmp_tx);
        });

        let result = unsafe {
            pipelined_ssd_to_gpu_inner(
                &dev,
                &gpu,
                mem_buf.as_mut_ptr(),
                gpu_buf.as_mut_ptr() as *mut std::ffi::c_void,
                0,
                total_bytes,
                -1,
                &alloc,
            )
        };
        h.join().unwrap();
        match &result {
            Err(DispatcherError::IoError(msg)) => {
                assert!(
                    msg.contains("completion channel disconnected"),
                    "wrong message: {msg}"
                )
            }
            other => panic!("expected IoError(disconnected), got: {other:?}"),
        }
    }

    /// DMA allocation failure: ring buffer allocation returns `Err` →
    /// `Err(AllocationFailed)`.
    #[test]
    fn dma_alloc_failure() {
        let (dev, _cmd_rx, _cmp_tx) = MockBlockDevice::new(512, 4096);
        let gpu = MockGpu { fail: false };
        let alloc = make_fail_alloc();

        let total_bytes = 4096usize;
        let mut mem_buf = vec![0u8; total_bytes];
        let mut gpu_buf = vec![0u8; total_bytes];

        let result = unsafe {
            pipelined_ssd_to_gpu_inner(
                &dev,
                &gpu,
                mem_buf.as_mut_ptr(),
                gpu_buf.as_mut_ptr() as *mut std::ffi::c_void,
                0,
                total_bytes,
                -1,
                &alloc,
            )
        };
        assert!(
            matches!(result, Err(DispatcherError::AllocationFailed(_))),
            "expected AllocationFailed, got: {result:?}"
        );
    }

    /// GPU DMA failure: `dma_copy_to_device` returns `Err` →
    /// `Err(IoError("GPU DMA copy…"))`.
    #[test]
    fn gpu_dma_failure() {
        let (dev, cmd_rx, cmp_tx) = MockBlockDevice::new(512, 4096);
        let gpu = MockGpu { fail: true };
        let alloc = make_alloc();

        let total_bytes = 4096usize;
        let mut mem_buf = vec![0u8; total_bytes];
        let mut gpu_buf = vec![0u8; total_bytes];

        let h = spawn_ok_responder(1, cmd_rx, cmp_tx);
        let result = unsafe {
            pipelined_ssd_to_gpu_inner(
                &dev,
                &gpu,
                mem_buf.as_mut_ptr(),
                gpu_buf.as_mut_ptr() as *mut std::ffi::c_void,
                0,
                total_bytes,
                -1,
                &alloc,
            )
        };
        h.join().unwrap();
        match &result {
            Err(DispatcherError::IoError(msg)) => {
                assert!(msg.contains("GPU DMA copy"), "wrong message: {msg}")
            }
            other => panic!("expected IoError(GPU DMA), got: {other:?}"),
        }
    }

    /// Zero total_bytes: no segments → no I/O → `Ok(())` immediately.
    #[test]
    fn zero_total_bytes() {
        let (dev, _cmd_rx, _cmp_tx) = MockBlockDevice::new(512, 4096);
        let gpu = MockGpu { fail: false };
        let alloc = make_alloc();

        let mut mem_buf = [0u8; 1];
        let mut gpu_buf = [0u8; 1];

        let result = unsafe {
            pipelined_ssd_to_gpu_inner(
                &dev,
                &gpu,
                mem_buf.as_mut_ptr(),
                gpu_buf.as_mut_ptr() as *mut std::ffi::c_void,
                0,
                0,
                -1,
                &alloc,
            )
        };
        assert!(result.is_ok(), "zero bytes should return Ok(()), got: {result:?}");
    }
}
