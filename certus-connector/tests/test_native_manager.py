# SPDX-License-Identifier: Apache-2.0
"""Unit tests for NativeCertusOffloadingManager.

Runs WITHOUT vllm, torch, or certus_native hardware — all dependencies mocked.
"""

import sys
import types
from dataclasses import dataclass
from typing import NewType
from unittest.mock import MagicMock

import pytest

# ── Mock vllm modules ─────────────────────────────────────────────────────
_mock_modules = {}
for mod_name in [
    "vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.kv_cache_utils",
    "vllm.v1.kv_offload", "vllm.v1.kv_offload.abstract",
    "vllm.v1.kv_offload.mediums", "vllm.v1.kv_offload.worker",
    "vllm.v1.kv_offload.worker.worker", "vllm.v1.kv_offload.spec",
    "vllm.v1.kv_cache_interface", "vllm.v1.attention",
    "vllm.v1.attention.backend", "vllm.config", "vllm.logger",
]:
    _mock_modules[mod_name] = types.ModuleType(mod_name)
    sys.modules[mod_name] = _mock_modules[mod_name]

from abc import ABC, abstractmethod

OffloadKey = NewType("OffloadKey", bytes)


@dataclass
class PrepareStoreOutput:
    keys_to_store: list
    store_spec: object
    evicted_keys: list


@dataclass
class OffloadingEvent:
    keys: list
    medium: str
    removed: bool


class LoadStoreSpec(ABC):
    @staticmethod
    @abstractmethod
    def medium() -> str: ...


class OffloadingManager(ABC):
    @abstractmethod
    def lookup(self, key, req_context=None): ...
    @abstractmethod
    def prepare_load(self, keys, req_context=None): ...
    def touch(self, keys): pass
    def complete_load(self, keys): pass
    @abstractmethod
    def prepare_store(self, keys, req_context=None): ...
    def complete_store(self, keys, success=True): pass
    def take_events(self): return ()
    def shutdown(self): pass


sys.modules["vllm.v1.kv_offload.abstract"].LoadStoreSpec = LoadStoreSpec
sys.modules["vllm.v1.kv_offload.abstract"].OffloadingManager = OffloadingManager
sys.modules["vllm.v1.kv_offload.abstract"].PrepareStoreOutput = PrepareStoreOutput
sys.modules["vllm.v1.kv_offload.abstract"].OffloadingEvent = OffloadingEvent
sys.modules["vllm.v1.kv_offload.abstract"].OffloadKey = OffloadKey
sys.modules["vllm.logger"].init_logger = lambda name: __import__("logging").getLogger(name)

# ── Mock certus_native before import ─────────────────────────────────────
_certus_native_mod = types.ModuleType("certus_native")
sys.modules["certus_native"] = _certus_native_mod

from certus_connector.native_manager import (  # noqa: E402
    NativeCertusOffloadingManager,
    _key_to_u64,
    _keys_to_u64s,
)
from certus_connector.mediums import BlockLocation, CertusLoadStoreSpec  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────

def make_key(block_hash: bytes, group_idx: int = 0) -> OffloadKey:
    return OffloadKey(block_hash + group_idx.to_bytes(4, "big"))


def simple_key(i: int) -> OffloadKey:
    return OffloadKey(i.to_bytes(8, "big") + (0).to_bytes(4, "big"))


class MockEngine:
    """Minimal stub of certus_native.CertusEngine."""

    def __init__(self):
        self._stored: dict = {}
        self.shutdown_called = False

    def batch_check(self, keys: list) -> int:
        return sum(1 for k in keys if self._stored.get(k))

    def prepare_store(self, keys: list):
        return (list(keys), [])

    def prepare_load(self, keys: list):
        return [(0xDEAD, 4096) for _ in keys]

    def touch(self, keys: list):
        pass

    def complete_load(self, keys: list):
        pass

    def complete_store(self, keys: list, success: bool):
        for k in keys:
            if success:
                self._stored[k] = True
            else:
                self._stored.pop(k, None)

    def shutdown(self):
        self.shutdown_called = True


# ── _key_to_u64 unit tests ────────────────────────────────────────────────

class TestKeyToU64:
    def test_eight_byte_key(self):
        key = bytes(range(8))
        result = _key_to_u64(key)
        expected = int.from_bytes(bytes(range(8)), "big")
        assert result == expected

    def test_short_key_uses_available_bytes(self):
        key = b"\x01\x02"
        result = _key_to_u64(key)
        assert result == int.from_bytes(b"\x01\x02", "big")

    def test_int_key_returned_as_is(self):
        assert _key_to_u64(42) == 42
        assert _key_to_u64(0) == 0

    def test_truncation_collision(self):
        """Document Bug 1: keys differing only in bytes 9+ produce the same u64."""
        prefix = b"\xAB\xCD\xEF\x01\x23\x45\x67\x89"
        key_group0 = prefix + (0).to_bytes(4, "big")
        key_group1 = prefix + (1).to_bytes(4, "big")
        assert key_group0 != key_group1
        assert _key_to_u64(key_group0) == _key_to_u64(key_group1)

    def test_full_twelve_byte_key(self):
        key = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B"
        result = _key_to_u64(key)
        assert result == int.from_bytes(key[:8], "big")

    def test_keys_to_u64s_maps_list(self):
        keys = [simple_key(i) for i in range(5)]
        results = _keys_to_u64s(keys)
        assert len(results) == 5
        assert all(isinstance(r, int) for r in results)


# ── NativeCertusOffloadingManager tests ──────────────────────────────────

class TestNativeCertusOffloadingManager:

    @pytest.fixture
    def engine(self):
        return MockEngine()

    @pytest.fixture
    def manager(self, engine):
        return NativeCertusOffloadingManager(engine)

    def test_lookup_hit(self, manager, engine):
        key = simple_key(1)
        int_key = _key_to_u64(key)
        engine._stored[int_key] = True
        result = manager.lookup(key)
        assert result is True

    def test_lookup_miss(self, manager):
        key = simple_key(99)
        result = manager.lookup(key)
        assert result is False

    def test_prepare_store_happy_path(self, manager):
        keys = [simple_key(i) for i in range(3)]
        output = manager.prepare_store(keys)
        assert output is not None
        assert len(output.keys_to_store) == 3
        assert output.evicted_keys == []

    def test_prepare_store_returns_none_when_engine_returns_none(self, manager, engine):
        engine.prepare_store = lambda keys: None
        output = manager.prepare_store([simple_key(0)])
        assert output is None

    def test_prepare_store_eviction_event_emitted(self, manager, engine):
        evicted_u64 = 0xBEEFCAFEDEAD0001
        engine.prepare_store = lambda keys: (list(keys), [evicted_u64])
        manager.prepare_store([simple_key(0)])
        events = list(manager.take_events())
        assert len(events) == 1
        assert events[0].removed is True
        evicted_key_bytes = evicted_u64.to_bytes(8, "big")
        assert evicted_key_bytes in events[0].keys

    def test_prepare_store_engine_returns_unexpected_key(self, manager, engine):
        """Bug 2: engine returns a u64 not in input list -> raises."""
        engine.prepare_store = lambda keys: ([999999999999], [])
        with pytest.raises((ValueError, KeyError, IndexError)):
            manager.prepare_store([simple_key(0)])

    def test_prepare_store_empty_input(self, manager):
        output = manager.prepare_store([])
        assert output is not None
        assert output.keys_to_store == []

    def test_take_events_clears_after_yield(self, manager, engine):
        evicted_u64 = 0xAAAA
        engine.prepare_store = lambda keys: (list(keys), [evicted_u64])
        manager.prepare_store([simple_key(0)])
        events1 = list(manager.take_events())
        assert len(events1) == 1
        events2 = list(manager.take_events())
        assert events2 == []

    def test_take_events_accumulates_across_calls(self, manager, engine):
        call_count = [0]
        evicted_u64 = 0xBBBB

        def mock_prepare_store(keys):
            call_count[0] += 1
            return (list(keys), [evicted_u64] if call_count[0] == 2 else [])

        engine.prepare_store = mock_prepare_store
        manager.prepare_store([simple_key(0)])
        manager.prepare_store([simple_key(1)])
        events = list(manager.take_events())
        assert len(events) == 1

    def test_complete_store_delegates_to_engine(self, manager, engine):
        called_with = []
        engine.complete_store = lambda keys, success: called_with.append((keys, success))
        manager.complete_store([simple_key(0)], success=True)
        assert len(called_with) == 1
        assert called_with[0][1] is True

    def test_complete_load_delegates_to_engine(self, manager, engine):
        called_with = []
        engine.complete_load = lambda keys: called_with.extend(keys)
        int_key = _key_to_u64(simple_key(7))
        manager.complete_load([simple_key(7)])
        assert int_key in called_with

    def test_shutdown_calls_engine_shutdown(self, manager, engine):
        manager.shutdown()
        assert engine.shutdown_called is True

    def test_import_error_raised_when_certus_native_missing(self):
        import certus_connector.native_manager as nm
        original = nm.certus_native
        nm.certus_native = None
        try:
            with pytest.raises(ImportError, match="certus_native is not installed"):
                NativeCertusOffloadingManager(MockEngine())
        finally:
            nm.certus_native = original
