use interfaces::ExtentManagerError;

use crate::block_io::BlockDeviceClient;
use crate::checkpoint::{self, SlabDescriptor};
use crate::error;
use crate::slab::{FREE_KEY, Slab};
use crate::superblock::{Superblock, SUPERBLOCK_SIZE};

pub(crate) type PerRegionData = Vec<Vec<SlabDescriptor>>;

pub(crate) fn recover(
    metadata_client: &BlockDeviceClient,
    component: &crate::ExtentManagerV2,
) -> Result<(Superblock, PerRegionData), ExtentManagerError> {
    let sb_data = metadata_client.read_blocks(0, SUPERBLOCK_SIZE)?;
    let sb = Superblock::deserialize(&sb_data)?;

    if sb.checkpoint_seq == 0 {
        let empty: PerRegionData = (0..sb.region_count as usize)
            .map(|_| Vec::new())
            .collect();
        return Ok((sb, empty));
    }

    let active_offset = sb.checkpoint_region_offset
        + sb.active_copy as u64 * sb.checkpoint_region_size;
    let inactive_offset = sb.checkpoint_region_offset
        + (1 - sb.active_copy) as u64 * sb.checkpoint_region_size;

    // Try active copy first
    match checkpoint::read_checkpoint_region(
        metadata_client,
        active_offset,
        sb.checkpoint_region_size,
        sb.checkpoint_seq,
    ) {
        Ok(data) => {
            let regions = checkpoint::deserialize_slabs(&data)?;
            return Ok((sb, regions));
        }
        Err(e) => {
            component.log_warn(&format!(
                "recovery_fallback: active checkpoint (copy {}) corrupt: {e}",
                sb.active_copy
            ));
        }
    }

    // Fall back to inactive copy (previous checkpoint)
    let prev_seq = sb.checkpoint_seq.saturating_sub(1);
    if prev_seq > 0 {
        match checkpoint::read_checkpoint_region(
            metadata_client,
            inactive_offset,
            sb.checkpoint_region_size,
            prev_seq,
        ) {
            Ok(data) => {
                let regions = checkpoint::deserialize_slabs(&data)?;
                return Ok((sb, regions));
            }
            Err(e) => {
                component.log_error(&format!(
                    "corruption_detected: both checkpoint copies corrupt: {e}"
                ));
            }
        }
    }

    Err(error::corrupt_metadata(
        "both active and inactive checkpoint copies are corrupt",
    ))
}

/// Reconstruct a `Slab` from a `SlabDescriptor` read from disk.
/// The allocation bitmap is derived from the key vector: any slot whose
/// key is not `FREE_KEY` is marked as allocated.
pub(crate) fn slab_from_descriptor(desc: &SlabDescriptor) -> Slab {
    let mut slab = Slab::new(desc.start_offset, desc.slab_size, desc.element_size);
    for (i, &key) in desc.keys.iter().enumerate() {
        if key != FREE_KEY {
            slab.mark_slot_allocated(i);
            slab.set_key(i, key);
        }
    }
    slab
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slab_from_fresh_descriptor_preserves_metadata() {
        let desc = SlabDescriptor {
            start_offset: 8_192,
            slab_size: 4 * 4_096,
            element_size: 4_096,
            keys: vec![FREE_KEY; 4],
        };

        let slab = slab_from_descriptor(&desc);

        assert_eq!(slab.start_offset, desc.start_offset);
        assert_eq!(slab.slab_size, desc.slab_size);
        assert_eq!(slab.element_size, desc.element_size);
        assert_eq!(slab.num_slots(), 4);
        assert!(slab.is_empty());
        assert_eq!(slab.bitmap.count_set(), 0);
        assert_eq!(slab.keys, desc.keys);
    }

    #[test]
    fn slab_from_descriptor_marks_only_non_free_keys_allocated() {
        let desc = SlabDescriptor {
            start_offset: 16_384,
            slab_size: 4 * 4_096,
            element_size: 4_096,
            keys: vec![11, FREE_KEY, 42, 0],
        };

        let slab = slab_from_descriptor(&desc);

        assert_eq!(slab.num_slots(), 4);
        assert!(!slab.is_empty());
        assert!(!slab.is_full());
        assert_eq!(slab.bitmap.count_set(), 3);
        assert_eq!(slab.get_key(0), 11);
        assert_eq!(slab.get_key(1), FREE_KEY);
        assert_eq!(slab.get_key(2), 42);
        assert_eq!(slab.get_key(3), 0);
    }

    #[test]
    #[should_panic]
    fn slab_from_corrupt_descriptor_panics_when_keys_exceed_capacity() {
        let desc = SlabDescriptor {
            start_offset: 0,
            slab_size: 4_096,
            element_size: 4_096,
            keys: vec![1, 2],
        };

        let _ = slab_from_descriptor(&desc);
    }
}
