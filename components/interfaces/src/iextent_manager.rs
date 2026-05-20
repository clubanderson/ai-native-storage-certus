//! Interface for the extent-manager component and shared types.
#[cfg(feature = "spdk")]
use component_macros::define_interface;
use std::fmt;

/// Opaque key identifying an extent.
///
/// # Examples
///
/// ```
/// use interfaces::ExtentKey;
///
/// let key: ExtentKey = 9;
/// assert_eq!(key, 9);
/// ```
pub type ExtentKey = u64;

/// A storage extent returned by the extent manager.
///
/// # Examples
///
/// ```
/// use interfaces::Extent;
///
/// let extent = Extent { key: 1, size: 128, offset: 4096 };
/// assert_eq!(extent.size, 128);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Extent {
    pub key: ExtentKey,
    pub size: u32, // size in blocks
    pub offset: u64,
}

/// Errors returned by `IExtentManager` operations.
///
/// # Examples
///
/// ```
/// use interfaces::ExtentManagerError;
///
/// let err = ExtentManagerError::OutOfSpace;
/// assert_eq!(err.to_string(), "out of space");
/// ```
#[derive(Debug, Clone)]
pub enum ExtentManagerError {
    CorruptMetadata(String),
    IoError(String),
    NotInitialized(String),
    OffsetNotFound(u64),
    OutOfSpace,
}

impl fmt::Display for ExtentManagerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CorruptMetadata(msg) => write!(f, "corrupt metadata: {msg}"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
            Self::NotInitialized(msg) => write!(f, "not initialized: {msg}"),
            Self::OffsetNotFound(off) => write!(f, "no extent at offset: {off}"),
            Self::OutOfSpace => write!(f, "out of space"),
        }
    }
}

impl std::error::Error for ExtentManagerError {}

/// Parameters used when formatting an extent-manager instance.
///
/// # Examples
///
/// ```
/// use interfaces::FormatParams;
///
/// let params = FormatParams::new(1 << 30, Some(7));
/// assert_eq!(params.data_disk_size, 1 << 30);
/// assert_eq!(params.instance_id, Some(7));
/// ```
#[derive(Debug, Clone)]
pub struct FormatParams {
    /// Total size of the data disk in bytes.
    pub data_disk_size: u64,
    /// Size of each slab in bytes. Must be a multiple of `sector_size`.
    pub slab_size: u64,
    /// Maximum extent size in bytes. Must be <= `slab_size`.
    pub max_extent_size: u32,
    /// Device sector size in bytes.
    pub sector_size: u32,
    /// Number of regions (must be a power of two).
    pub region_count: u32,
    /// Alignment of checkpoint regions on the metadata disk.
    /// The first checkpoint region starts at the first multiple of this
    /// value that is >= the superblock size.
    pub metadata_alignment: u64,
    /// Instance identifier stored in the superblock. If None, a random
    /// value is generated at format time.
    pub instance_id: Option<u64>,
    /// NVMe namespace identifier for the metadata disk.
    pub metadata_disk_ns_id: u32,
}

impl FormatParams {
    /// Create format parameters with sensible defaults for a new data disk.
    ///
    /// # Examples
    ///
    /// ```
    /// use interfaces::FormatParams;
    ///
    /// let params = FormatParams::new(8 << 30, None);
    /// assert_eq!(params.data_disk_size, 8 << 30);
    /// ```
    pub fn new(data_disk_size: u64, instance_id: Option<u64>) -> Self {
        Self {
            data_disk_size,
            instance_id,
            ..Default::default()
        }
    }
}

impl Default for FormatParams {
    fn default() -> Self {
        Self {
            data_disk_size: 0,
            slab_size: 1024 * 1024 * 1024,       // 1 GiB
            max_extent_size: 1024 * 1024 * 1024, // 1 GiB
            sector_size: 4096,                   // 4 KiB
            region_count: 16,
            metadata_alignment: 128 * 1024, // 128 KiB
            instance_id: None,
            metadata_disk_ns_id: 1,
        }
    }
}

/// Tracks a reserved extent until it is either published or aborted.
///
/// Dropping an unpublished handle automatically aborts the reservation.
///
/// # Examples
///
/// ```
/// use interfaces::{Extent, WriteHandle};
///
/// let handle = WriteHandle::new(
///     5,
///     8192,
///     32,
///     Box::new(|| Ok(Extent { key: 5, offset: 8192, size: 32 })),
///     Box::new(|| {}),
/// );
/// assert_eq!(handle.key(), 5);
/// ```
pub struct WriteHandle {
    key: ExtentKey,
    offset: u64,
    size: u32,
    publish_fn: Option<Box<dyn FnOnce() -> Result<Extent, ExtentManagerError> + Send>>,
    abort_fn: Option<Box<dyn FnOnce() + Send>>,
}

impl WriteHandle {
    /// Create a new write handle for a reserved extent.
    ///
    /// # Examples
    ///
    /// ```
    /// use interfaces::{Extent, WriteHandle};
    ///
    /// let handle = WriteHandle::new(
    ///     1,
    ///     4096,
    ///     16,
    ///     Box::new(|| Ok(Extent { key: 1, offset: 4096, size: 16 })),
    ///     Box::new(|| {}),
    /// );
    /// assert_eq!(handle.extent_offset(), 4096);
    /// ```
    pub fn new(
        key: ExtentKey,
        offset: u64,
        size: u32,
        publish_fn: Box<dyn FnOnce() -> Result<Extent, ExtentManagerError> + Send>,
        abort_fn: Box<dyn FnOnce() + Send>,
    ) -> Self {
        Self {
            key,
            offset,
            size,
            publish_fn: Some(publish_fn),
            abort_fn: Some(abort_fn),
        }
    }

    /// Return the key associated with this reservation.
    ///
    /// # Examples
    ///
    /// ```
    /// use interfaces::{Extent, WriteHandle};
    ///
    /// let handle = WriteHandle::new(
    ///     3,
    ///     12288,
    ///     8,
    ///     Box::new(|| Ok(Extent { key: 3, offset: 12288, size: 8 })),
    ///     Box::new(|| {}),
    /// );
    /// assert_eq!(handle.key(), 3);
    /// ```
    pub fn key(&self) -> ExtentKey {
        self.key
    }

    /// Return the starting offset of the reserved extent.
    ///
    /// # Examples
    ///
    /// ```
    /// use interfaces::{Extent, WriteHandle};
    ///
    /// let handle = WriteHandle::new(
    ///     4,
    ///     16384,
    ///     4,
    ///     Box::new(|| Ok(Extent { key: 4, offset: 16384, size: 4 })),
    ///     Box::new(|| {}),
    /// );
    /// assert_eq!(handle.extent_offset(), 16384);
    /// ```
    pub fn extent_offset(&self) -> u64 {
        self.offset
    }

    /// Return the size of the reserved extent in blocks.
    ///
    /// # Examples
    ///
    /// ```
    /// use interfaces::{Extent, WriteHandle};
    ///
    /// let handle = WriteHandle::new(
    ///     6,
    ///     20480,
    ///     64,
    ///     Box::new(|| Ok(Extent { key: 6, offset: 20480, size: 64 })),
    ///     Box::new(|| {}),
    /// );
    /// assert_eq!(handle.extent_size(), 64);
    /// ```
    pub fn extent_size(&self) -> u32 {
        self.size
    }

    /// Publish the reserved extent and return its committed metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use interfaces::{Extent, WriteHandle};
    ///
    /// let handle = WriteHandle::new(
    ///     8,
    ///     24576,
    ///     2,
    ///     Box::new(|| Ok(Extent { key: 8, offset: 24576, size: 2 })),
    ///     Box::new(|| {}),
    /// );
    /// let extent = handle.publish().unwrap();
    /// assert_eq!(extent.key, 8);
    /// ```
    pub fn publish(mut self) -> Result<Extent, ExtentManagerError> {
        let f = self
            .publish_fn
            .take()
            .expect("publish called on consumed handle");
        self.abort_fn.take();
        f()
    }

    /// Abort the reservation without publishing it.
    ///
    /// # Examples
    ///
    /// ```
    /// use interfaces::{Extent, WriteHandle};
    ///
    /// let handle = WriteHandle::new(
    ///     10,
    ///     28672,
    ///     1,
    ///     Box::new(|| Ok(Extent { key: 10, offset: 28672, size: 1 })),
    ///     Box::new(|| {}),
    /// );
    /// handle.abort();
    /// ```
    pub fn abort(mut self) {
        self.publish_fn.take();
        if let Some(f) = self.abort_fn.take() {
            f();
        }
    }
}

impl Drop for WriteHandle {
    fn drop(&mut self) {
        if let Some(f) = self.abort_fn.take() {
            f();
        }
    }
}

impl fmt::Debug for WriteHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriteHandle")
            .field("key", &self.key)
            .field("offset", &self.offset)
            .field("size", &self.size)
            .field("has_publish_fn", &self.publish_fn.is_some())
            .field("has_abort_fn", &self.abort_fn.is_some())
            .finish()
    }
}

/// Allocates, publishes, and checkpoints extents on persistent storage.
///
/// # Examples
///
/// ```no_run
/// use interfaces::IExtentManager;
///
/// fn checkpoint(manager: &dyn IExtentManager) {
///     manager.checkpoint().unwrap();
/// }
/// ```
#[cfg(feature = "spdk")]
define_interface! {
    pub IExtentManager {
        fn format(&self, params: FormatParams) -> Result<(), ExtentManagerError>;

        fn initialize(&self) -> Result<(), ExtentManagerError>;

        fn reserve_extent(
            &self,
            key: ExtentKey,
            size: u32,
        ) -> Result<WriteHandle, ExtentManagerError>;

        fn get_extents(&self) -> Vec<Extent>;

        fn for_each_extent(&self, cb: &mut dyn FnMut(&Extent));

        fn remove_extent(&self, offset: u64) -> Result<(), ExtentManagerError>;

        fn checkpoint(&self) -> Result<(), ExtentManagerError>;

        fn get_instance_id(&self) -> Result<u64, ExtentManagerError>;

        /// Set the automatic checkpoint interval.
        ///
        /// `Some(duration)` enables the background checkpoint thread to fire
        /// every `duration`. `None` disables automatic checkpoints entirely;
        /// callers must then invoke `checkpoint()` manually. The default is
        /// five minutes.
        fn set_checkpoint_interval(&self, interval: Option<std::time::Duration>);

        /// Return the number of bytes currently allocated across all regions.
        fn used_bytes(&self) -> u64;

        /// Return the total usable capacity in bytes across all regions.
        fn capacity_bytes(&self) -> u64;
    }
}
