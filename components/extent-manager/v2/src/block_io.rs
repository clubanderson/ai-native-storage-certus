use interfaces::{
    ClientChannels, Command, Completion, DmaAllocFn, DmaBuffer, NvmeBlockError,
};
use std::sync::{Arc, Mutex};

use crate::error;

pub(crate) struct BlockDeviceClient {
    channels: ClientChannels,
    alloc: DmaAllocFn,
    sector_size: u32,
    ns_id: u32,
}

impl BlockDeviceClient {
    pub fn new(channels: ClientChannels, alloc: DmaAllocFn, sector_size: u32, ns_id: u32) -> Self {
        Self {
            channels,
            alloc,
            sector_size,
            ns_id,
        }
    }

    pub fn alloc_buffer(&self, size: usize) -> Result<DmaBuffer, NvmeBlockError> {
        let align = self.sector_size as usize;
        (self.alloc)(size, align, None).map_err(|e| {
            NvmeBlockError::BlockDevice(interfaces::BlockDeviceError::DmaAllocationFailed(e))
        })
    }

    pub fn write_blocks(
        &self,
        lba: u64,
        data: &[u8],
    ) -> Result<(), interfaces::ExtentManagerError> {
        let num_blocks =
            (data.len() + self.sector_size as usize - 1) / self.sector_size as usize;
        let buf_size = num_blocks * self.sector_size as usize;

        let mut buf = self.alloc_buffer(buf_size).map_err(error::nvme_to_em)?;
        buf.as_mut_slice()[..data.len()].copy_from_slice(data);
        if data.len() < buf_size {
            for b in &mut buf.as_mut_slice()[data.len()..buf_size] {
                *b = 0;
            }
        }

        #[allow(clippy::arc_with_non_send_sync)]
        let buf = Arc::new(buf);

        for i in 0..num_blocks {
            let block_lba = lba + i as u64;
            let block_start = i * self.sector_size as usize;
            let block_end = block_start + self.sector_size as usize;

            let mut block_buf = self
                .alloc_buffer(self.sector_size as usize)
                .map_err(error::nvme_to_em)?;
            block_buf.as_mut_slice()[..self.sector_size as usize]
                .copy_from_slice(&buf.as_slice()[block_start..block_end]);

            #[allow(clippy::arc_with_non_send_sync)]
            let block_buf = Arc::new(block_buf);

            self.channels
                .command_tx
                .send(Command::WriteSync {
                    ns_id: self.ns_id,
                    lba: block_lba,
                    buf: block_buf,
                })
                .map_err(|_| error::io_error("write command send failed"))?;

            match self.channels.completion_rx.recv() {
                Ok(Completion::WriteDone { result, .. }) => {
                    result.map_err(error::nvme_to_em)?;
                }
                Ok(Completion::Error { error: e, .. }) => {
                    return Err(error::nvme_to_em(e));
                }
                Ok(_) => {
                    return Err(error::io_error("unexpected completion type"));
                }
                Err(_) => {
                    return Err(error::io_error("write completion recv failed"));
                }
            }
        }

        Ok(())
    }

    pub fn read_blocks(
        &self,
        lba: u64,
        num_bytes: usize,
    ) -> Result<Vec<u8>, interfaces::ExtentManagerError> {
        let num_blocks =
            (num_bytes + self.sector_size as usize - 1) / self.sector_size as usize;
        let mut result = Vec::with_capacity(num_bytes);

        for i in 0..num_blocks {
            let block_lba = lba + i as u64;
            let buf = self
                .alloc_buffer(self.sector_size as usize)
                .map_err(error::nvme_to_em)?;
            let buf = Arc::new(Mutex::new(buf));

            self.channels
                .command_tx
                .send(Command::ReadSync {
                    ns_id: self.ns_id,
                    lba: block_lba,
                    buf: Arc::clone(&buf),
                })
                .map_err(|_| error::io_error("read command send failed"))?;

            match self.channels.completion_rx.recv() {
                Ok(Completion::ReadDone { result: res, .. }) => {
                    res.map_err(error::nvme_to_em)?;
                    let locked = buf.lock().unwrap();
                    let remaining = num_bytes - result.len();
                    let to_copy = remaining.min(self.sector_size as usize);
                    result.extend_from_slice(&locked.as_slice()[..to_copy]);
                }
                Ok(Completion::Error { error: e, .. }) => {
                    return Err(error::nvme_to_em(e));
                }
                Ok(_) => {
                    return Err(error::io_error("unexpected completion type"));
                }
                Err(_) => {
                    return Err(error::io_error("read completion recv failed"));
                }
            }
        }

        Ok(result)
    }

    pub fn sector_size(&self) -> u32 {
        self.sector_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use component_core::channel::SpscChannel;
    use interfaces::{BlockDeviceError, ClientChannels, Command, Completion, NvmeBlockError, OpHandle};
    use std::sync::Arc;
    use std::thread;

    use crate::test_support::heap_dma_alloc;

    const SECTOR_SIZE: u32 = 512;
    const NS_ID: u32 = 1;

    /// Create paired channels for a test client.
    ///
    /// Returns:
    /// - `ClientChannels` to hand to `BlockDeviceClient`
    /// - `Sender<Completion>` so the test thread can inject completions
    /// - `Receiver<Command>` so the test thread can inspect sent commands
    fn make_channels() -> (
        ClientChannels,
        component_core::channel::Sender<Completion>,
        component_core::channel::Receiver<Command>,
    ) {
        let cmd_ch = SpscChannel::<Command>::new(64);
        let cmd_tx = cmd_ch.sender().unwrap();
        let cmd_rx = cmd_ch.receiver().unwrap();

        let comp_ch = SpscChannel::<Completion>::new(64);
        let comp_tx = comp_ch.sender().unwrap();
        let comp_rx = comp_ch.receiver().unwrap();

        let channels = ClientChannels {
            command_tx: cmd_tx,
            completion_rx: comp_rx,
        };
        (channels, comp_tx, cmd_rx)
    }

    fn failing_alloc() -> DmaAllocFn {
        Arc::new(|_size, _align, _numa| Err("simulated alloc failure".to_string()))
    }

    // -----------------------------------------------------------------------
    // alloc_buffer
    // -----------------------------------------------------------------------

    #[test]
    fn alloc_buffer_happy_path() {
        let (channels, _comp_tx, _cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, heap_dma_alloc(), SECTOR_SIZE, NS_ID);
        let buf = client.alloc_buffer(SECTOR_SIZE as usize).expect("allocation should succeed");
        assert_eq!(buf.len(), SECTOR_SIZE as usize);
    }

    #[test]
    fn alloc_buffer_forwards_size_and_align_to_allocator() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let captured_size = Arc::new(AtomicUsize::new(0));
        let captured_align = Arc::new(AtomicUsize::new(0));
        let cs = Arc::clone(&captured_size);
        let ca = Arc::clone(&captured_align);

        let alloc: DmaAllocFn = Arc::new(move |size, align, _numa| {
            cs.store(size, Ordering::Relaxed);
            ca.store(align, Ordering::Relaxed);
            Err("capture done".to_string())
        });

        let (channels, _comp_tx, _cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, alloc, SECTOR_SIZE, NS_ID);
        let _ = client.alloc_buffer(1024);

        assert_eq!(captured_size.load(Ordering::Relaxed), 1024);
        assert_eq!(captured_align.load(Ordering::Relaxed), SECTOR_SIZE as usize);
    }

    #[test]
    fn alloc_buffer_error_string_preserved_in_dma_allocation_failed() {
        let (channels, _comp_tx, _cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, failing_alloc(), SECTOR_SIZE, NS_ID);
        let err = client.alloc_buffer(SECTOR_SIZE as usize).unwrap_err();
        assert!(
            matches!(
                err,
                NvmeBlockError::BlockDevice(BlockDeviceError::DmaAllocationFailed(_))
            ),
            "expected DmaAllocationFailed, got: {err:?}"
        );
        assert!(
            err.to_string().contains("simulated alloc failure"),
            "error string not preserved: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // write_blocks
    // -----------------------------------------------------------------------

    #[test]
    fn write_blocks_happy_path() {
        let data = b"hello world!";
        let (channels, comp_tx, cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, heap_dma_alloc(), SECTOR_SIZE, NS_ID);

        let handle = thread::spawn(move || {
            let _cmd = cmd_rx.recv().expect("should receive WriteSync");
            comp_tx
                .send(Completion::WriteDone {
                    handle: OpHandle(1),
                    result: Ok(()),
                })
                .unwrap();
        });

        let result = client.write_blocks(0, data);
        handle.join().unwrap();
        assert!(result.is_ok(), "expected Ok, got: {result:?}");
    }

    #[test]
    fn write_blocks_alloc_failure_returns_io_error() {
        let (channels, _comp_tx, _cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, failing_alloc(), SECTOR_SIZE, NS_ID);
        let result = client.write_blocks(0, b"any data");
        assert!(result.is_err(), "expected Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("simulated alloc failure"),
            "expected alloc error message, got: {msg}"
        );
    }

    #[test]
    fn write_blocks_device_error_propagated() {
        let data = vec![0u8; SECTOR_SIZE as usize];
        let (channels, comp_tx, cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, heap_dma_alloc(), SECTOR_SIZE, NS_ID);

        let handle = thread::spawn(move || {
            let _cmd = cmd_rx.recv().unwrap();
            comp_tx
                .send(Completion::WriteDone {
                    handle: OpHandle(1),
                    result: Err(NvmeBlockError::BlockDevice(BlockDeviceError::WriteFailed(
                        "disk is dead".into(),
                    ))),
                })
                .unwrap();
        });

        let result = client.write_blocks(0, &data);
        handle.join().unwrap();
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("disk is dead"),
            "device error message not propagated"
        );
    }

    #[test]
    fn write_blocks_error_completion_propagated() {
        let data = vec![0u8; SECTOR_SIZE as usize];
        let (channels, comp_tx, cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, heap_dma_alloc(), SECTOR_SIZE, NS_ID);

        let handle = thread::spawn(move || {
            let _cmd = cmd_rx.recv().unwrap();
            comp_tx
                .send(Completion::Error {
                    handle: Some(OpHandle(1)),
                    error: NvmeBlockError::BlockDevice(BlockDeviceError::WriteFailed(
                        "device error path".into(),
                    )),
                })
                .unwrap();
        });

        let result = client.write_blocks(0, &data);
        handle.join().unwrap();
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("device error path"),
            "error completion not propagated"
        );
    }

    #[test]
    fn write_blocks_unexpected_completion_returns_error() {
        let data = vec![0u8; SECTOR_SIZE as usize];
        let (channels, comp_tx, cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, heap_dma_alloc(), SECTOR_SIZE, NS_ID);

        let handle = thread::spawn(move || {
            let _cmd = cmd_rx.recv().unwrap();
            // Send WriteZerosDone instead of WriteDone
            comp_tx
                .send(Completion::WriteZerosDone {
                    handle: OpHandle(1),
                    result: Ok(()),
                })
                .unwrap();
        });

        let result = client.write_blocks(0, &data);
        handle.join().unwrap();
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("unexpected completion"),
            "expected 'unexpected completion' error"
        );
    }

    #[test]
    fn write_blocks_completion_recv_closed_returns_io_error() {
        let data = vec![0u8; SECTOR_SIZE as usize];
        let (channels, comp_tx, cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, heap_dma_alloc(), SECTOR_SIZE, NS_ID);

        let handle = thread::spawn(move || {
            // Consume the command then close the completion channel
            let _cmd = cmd_rx.recv().unwrap();
            drop(comp_tx);
        });

        let result = client.write_blocks(0, &data);
        handle.join().unwrap();
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("recv failed"),
            "expected recv-failed error"
        );
    }

    #[test]
    fn write_blocks_multi_block_increments_lba() {
        // Two sectors of data → two WriteSync commands with consecutive LBAs
        let data = vec![0xABu8; SECTOR_SIZE as usize * 2];
        let (channels, comp_tx, cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, heap_dma_alloc(), SECTOR_SIZE, NS_ID);

        let handle = thread::spawn(move || {
            let mut lbas = Vec::new();
            for _ in 0..2 {
                match cmd_rx.recv().unwrap() {
                    Command::WriteSync { lba, .. } => {
                        lbas.push(lba);
                        comp_tx
                            .send(Completion::WriteDone {
                                handle: OpHandle(1),
                                result: Ok(()),
                            })
                            .unwrap();
                    }
                    _ => panic!("expected WriteSync"),
                }
            }
            lbas
        });

        let result = client.write_blocks(5, &data);
        let lbas = handle.join().unwrap();
        assert!(result.is_ok());
        assert_eq!(lbas, vec![5u64, 6u64]);
    }

    // -----------------------------------------------------------------------
    // read_blocks
    // -----------------------------------------------------------------------

    #[test]
    fn read_blocks_happy_path_returns_data() {
        let payload = b"hello read!";
        let (channels, comp_tx, cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, heap_dma_alloc(), SECTOR_SIZE, NS_ID);

        let expected = payload.to_vec();
        let handle = thread::spawn(move || {
            match cmd_rx.recv().unwrap() {
                Command::ReadSync { buf, .. } => {
                    let mut locked = buf.lock().unwrap();
                    locked.as_mut_slice()[..expected.len()].copy_from_slice(&expected);
                }
                _ => panic!("expected ReadSync"),
            }
            comp_tx
                .send(Completion::ReadDone {
                    handle: OpHandle(1),
                    result: Ok(()),
                })
                .unwrap();
        });

        let result = client.read_blocks(0, payload.len());
        handle.join().unwrap();
        let data = result.expect("expected Ok");
        assert_eq!(&data[..payload.len()], payload);
    }

    #[test]
    fn read_blocks_alloc_failure_returns_error() {
        let (channels, _comp_tx, _cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, failing_alloc(), SECTOR_SIZE, NS_ID);
        let result = client.read_blocks(0, 128);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("simulated alloc failure"),
            "expected alloc error message"
        );
    }

    #[test]
    fn read_blocks_device_error_propagated() {
        let (channels, comp_tx, cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, heap_dma_alloc(), SECTOR_SIZE, NS_ID);

        let handle = thread::spawn(move || {
            let _cmd = cmd_rx.recv().unwrap();
            comp_tx
                .send(Completion::ReadDone {
                    handle: OpHandle(1),
                    result: Err(NvmeBlockError::BlockDevice(BlockDeviceError::ReadFailed(
                        "disk read error".into(),
                    ))),
                })
                .unwrap();
        });

        let result = client.read_blocks(0, SECTOR_SIZE as usize);
        handle.join().unwrap();
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("disk read error"),
            "device read error not propagated"
        );
    }

    #[test]
    fn read_blocks_error_completion_propagated() {
        let (channels, comp_tx, cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, heap_dma_alloc(), SECTOR_SIZE, NS_ID);

        let handle = thread::spawn(move || {
            let _cmd = cmd_rx.recv().unwrap();
            comp_tx
                .send(Completion::Error {
                    handle: Some(OpHandle(1)),
                    error: NvmeBlockError::BlockDevice(BlockDeviceError::ReadFailed(
                        "error completion on read".into(),
                    )),
                })
                .unwrap();
        });

        let result = client.read_blocks(0, SECTOR_SIZE as usize);
        handle.join().unwrap();
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("error completion on read"),
            "error completion not propagated"
        );
    }

    #[test]
    fn read_blocks_unexpected_completion_returns_error() {
        let (channels, comp_tx, cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, heap_dma_alloc(), SECTOR_SIZE, NS_ID);

        let handle = thread::spawn(move || {
            let _cmd = cmd_rx.recv().unwrap();
            // Send WriteDone instead of ReadDone
            comp_tx
                .send(Completion::WriteDone {
                    handle: OpHandle(1),
                    result: Ok(()),
                })
                .unwrap();
        });

        let result = client.read_blocks(0, SECTOR_SIZE as usize);
        handle.join().unwrap();
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("unexpected completion"),
            "expected 'unexpected completion' error"
        );
    }

    #[test]
    fn read_blocks_completion_recv_closed_returns_io_error() {
        let (channels, comp_tx, cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, heap_dma_alloc(), SECTOR_SIZE, NS_ID);

        let handle = thread::spawn(move || {
            let _cmd = cmd_rx.recv().unwrap();
            drop(comp_tx);
        });

        let result = client.read_blocks(0, SECTOR_SIZE as usize);
        handle.join().unwrap();
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("recv failed"),
            "expected recv-failed error"
        );
    }

    #[test]
    fn read_blocks_returns_exact_requested_byte_count() {
        // Request 3 bytes from a full-sector read — result must be exactly 3 bytes
        let read_len = 3usize;
        let (channels, comp_tx, cmd_rx) = make_channels();
        let client = BlockDeviceClient::new(channels, heap_dma_alloc(), SECTOR_SIZE, NS_ID);

        let handle = thread::spawn(move || {
            match cmd_rx.recv().unwrap() {
                Command::ReadSync { buf, .. } => {
                    let mut locked = buf.lock().unwrap();
                    locked.as_mut_slice()[..3].copy_from_slice(b"abc");
                }
                _ => panic!("expected ReadSync"),
            }
            comp_tx
                .send(Completion::ReadDone {
                    handle: OpHandle(1),
                    result: Ok(()),
                })
                .unwrap();
        });

        let result = client.read_blocks(0, read_len);
        handle.join().unwrap();
        let data = result.expect("expected Ok");
        assert_eq!(data.len(), read_len, "byte count mismatch");
        assert_eq!(&data, b"abc");
    }
}
