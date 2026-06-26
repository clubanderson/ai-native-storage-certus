# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SimpleLRUTarget."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from replay_offloading_traces import PrepareStoreOutput, SimpleLRUTarget


def make_key(value: int) -> bytes:
    return f"block-{value}".encode()


class TestSimpleLRUTarget:
    @pytest.fixture
    def target(self):
        return SimpleLRUTarget(num_blocks=3)

    def _store(self, target: SimpleLRUTarget, *keys: bytes, success: bool = True):
        output = target.prepare_store(list(keys))
        assert output is not None
        target.complete_store(output.block_hashes_to_store, success=success)
        return output

    def test_init_records_capacity_and_default_block_size(self):
        target = SimpleLRUTarget(num_blocks=4)
        assert target.capacity == 4
        assert target.block_size == 16
        assert list(target._cache.keys()) == []
        assert target._pending == set()

    def test_init_accepts_custom_block_size(self):
        target = SimpleLRUTarget(num_blocks=2, block_size=64)
        assert target.block_size == 64

    def test_lookup_empty_cache_returns_zero(self, target):
        assert target.lookup([make_key(1)]) == 0

    def test_store_and_lookup_single_key(self, target):
        self._store(target, make_key(1))
        assert target.lookup([make_key(1)]) == 1

    def test_store_and_lookup_multiple_keys(self, target):
        self._store(target, make_key(1), make_key(2), make_key(3))
        assert target.lookup([make_key(1), make_key(2), make_key(3)]) == 3

    def test_lookup_prefix_match_stops_at_first_miss(self, target):
        self._store(target, make_key(1), make_key(2), make_key(3))
        assert target.lookup([make_key(1), make_key(2), make_key(99), make_key(3)]) == 2

    def test_lookup_all_miss_returns_zero(self, target):
        self._store(target, make_key(1))
        assert target.lookup([make_key(2), make_key(3)]) == 0

    def test_lookup_partial_hit_counts_prefix_only(self, target):
        self._store(target, make_key(1), make_key(3))
        assert target.lookup([make_key(1), make_key(2), make_key(3)]) == 1

    def test_touch_present_key_moves_to_end(self, target):
        self._store(target, make_key(1), make_key(2), make_key(3))
        target.touch([make_key(1)])
        assert list(target._cache.keys()) == [make_key(2), make_key(3), make_key(1)]

    def test_touch_absent_key_is_noop(self, target):
        self._store(target, make_key(1), make_key(2))
        before = list(target._cache.keys())
        target.touch([make_key(99)])
        assert list(target._cache.keys()) == before

    def test_prepare_load_present_key_succeeds(self, target):
        self._store(target, make_key(1))
        target.prepare_load([make_key(1)])

    def test_prepare_load_updates_recency(self, target):
        self._store(target, make_key(1), make_key(2), make_key(3))
        target.prepare_load([make_key(1)])
        assert list(target._cache.keys())[-1] == make_key(1)

    def test_prepare_load_missing_key_raises_keyerror(self, target):
        with pytest.raises(KeyError, match="prepare_load miss"):
            target.prepare_load([make_key(1)])

    def test_complete_load_returns_none(self, target):
        assert target.complete_load([make_key(1)]) is None

    def test_prepare_store_returns_to_store_and_evicted_lists(self, target):
        self._store(target, make_key(1), make_key(2), make_key(3))
        output = target.prepare_store([make_key(4)])
        assert output.block_hashes_to_store == [make_key(4)]
        assert output.block_hashes_evicted == [make_key(1)]

    def test_prepare_store_returns_none_when_request_exceeds_capacity(self, target):
        assert target.prepare_store([make_key(1), make_key(2), make_key(3), make_key(4)]) is None

    def test_prepare_store_skips_already_cached_keys(self, target):
        self._store(target, make_key(1), make_key(2))
        output = target.prepare_store([make_key(1), make_key(2), make_key(3)])
        assert output.block_hashes_to_store == [make_key(3)]

    def test_prepare_store_skips_already_pending_keys(self, target):
        target.prepare_store([make_key(1), make_key(2)])
        output = target.prepare_store([make_key(1), make_key(2), make_key(3)])
        assert output.block_hashes_to_store == [make_key(3)]

    def test_prepare_store_with_only_cached_and_pending_keys_returns_empty(self, target):
        self._store(target, make_key(1))
        target.prepare_store([make_key(2)])
        output = target.prepare_store([make_key(1), make_key(2)])
        assert output.block_hashes_to_store == []
        assert output.block_hashes_evicted == []

    def test_complete_store_success_adds_keys_to_cache(self, target):
        output = target.prepare_store([make_key(1), make_key(2)])
        target.complete_store(output.block_hashes_to_store, success=True)
        assert list(target._cache.keys()) == [make_key(1), make_key(2)]

    def test_complete_store_failure_does_not_add_keys_to_cache(self, target):
        output = target.prepare_store([make_key(1), make_key(2)])
        target.complete_store(output.block_hashes_to_store, success=False)
        assert list(target._cache.keys()) == []

    def test_eviction_removes_oldest_key_first(self, target):
        self._store(target, make_key(1), make_key(2), make_key(3))
        output = target.prepare_store([make_key(4)])
        assert output.block_hashes_evicted == [make_key(1)]
        assert list(target._cache.keys()) == [make_key(2), make_key(3)]

    def test_eviction_can_remove_multiple_keys_in_lru_order(self):
        target = SimpleLRUTarget(num_blocks=2)
        self._store(target, make_key(1), make_key(2))
        output = target.prepare_store([make_key(3), make_key(4)])
        assert output.block_hashes_evicted == [make_key(1), make_key(2)]

    def test_eviction_skips_pending_keys(self):
        target = SimpleLRUTarget(num_blocks=2)
        self._store(target, make_key(1))
        target.prepare_store([make_key(2)])
        output = target.prepare_store([make_key(3)])
        assert output.block_hashes_evicted == [make_key(1)]
        assert target._pending == {make_key(2), make_key(3)}

    def test_complete_store_clears_pending_on_success(self, target):
        output = target.prepare_store([make_key(1)])
        target.complete_store(output.block_hashes_to_store, success=True)
        assert target._pending == set()

    def test_complete_store_clears_pending_on_failure(self, target):
        output = target.prepare_store([make_key(1)])
        target.complete_store(output.block_hashes_to_store, success=False)
        assert target._pending == set()

    def test_zero_capacity_rejects_nonempty_store(self):
        target = SimpleLRUTarget(num_blocks=0)
        assert target.prepare_store([make_key(1)]) is None

    def test_zero_capacity_accepts_empty_store(self):
        target = SimpleLRUTarget(num_blocks=0)
        output = target.prepare_store([])
        assert isinstance(output, PrepareStoreOutput)
        assert output.block_hashes_to_store == []
        assert output.block_hashes_evicted == []

    def test_duplicate_keys_in_prepare_store_are_preserved(self):
        target = SimpleLRUTarget(num_blocks=4)
        output = target.prepare_store([make_key(1), make_key(1), make_key(2)])
        assert output.block_hashes_to_store == [make_key(1), make_key(1), make_key(2)]
        target.complete_store(output.block_hashes_to_store, success=True)
        assert list(target._cache.keys()) == [make_key(1), make_key(2)]

    def test_capacity_one_evicts_existing_key(self):
        target = SimpleLRUTarget(num_blocks=1)
        self._store(target, make_key(1))
        output = target.prepare_store([make_key(2)])
        assert output.block_hashes_evicted == [make_key(1)]

    def test_touch_changes_future_eviction_choice(self, target):
        self._store(target, make_key(1), make_key(2), make_key(3))
        target.touch([make_key(1)])
        output = target.prepare_store([make_key(4)])
        assert output.block_hashes_evicted == [make_key(2)]
