# SPDX-License-Identifier: Apache-2.0
"""Unit tests for NativeCertusOffloadingManager.

Runs without vLLM or certus_native installed by mocking imports via sys.modules.
"""

from __future__ import annotations

import importlib
import sys
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import NewType
from unittest.mock import Mock

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

OffloadKey = NewType("OffloadKey", bytes)


def make_key(value: int, suffix: int = 0) -> OffloadKey:
    return OffloadKey(value.to_bytes(8, "big") + suffix.to_bytes(4, "big"))


def install_vllm_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_modules = {}
    for mod_name in [
        "vllm",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.kv_cache_utils",
        "vllm.v1.kv_offload",
        "vllm.v1.kv_offload.abstract",
        "vllm.v1.kv_offload.mediums",
        "vllm.v1.kv_offload.worker",
        "vllm.v1.kv_offload.worker.worker",
        "vllm.v1.kv_offload.spec",
        "vllm.v1.kv_cache_interface",
        "vllm.v1.attention",
        "vllm.v1.attention.backend",
        "vllm.config",
        "vllm.logger",
    ]:
        module = types.ModuleType(mod_name)
        mock_modules[mod_name] = module
        monkeypatch.setitem(sys.modules, mod_name, module)

    class LoadStoreSpec(ABC):
        @staticmethod
        @abstractmethod
        def medium() -> str: ...

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

    class OffloadingManager(ABC):
        @abstractmethod
        def lookup(self, key, req_context=None): ...

        @abstractmethod
        def prepare_load(self, keys, req_context=None): ...

        def touch(self, keys):
            return None

        def complete_load(self, keys):
            return None

        @abstractmethod
        def prepare_store(self, keys, req_context=None): ...

        def complete_store(self, keys, success=True):
            return None

        def take_events(self):
            return ()

        def shutdown(self):
            return None

    abstract = mock_modules["vllm.v1.kv_offload.abstract"]
    abstract.LoadStoreSpec = LoadStoreSpec
    abstract.OffloadingManager = OffloadingManager
    abstract.OffloadKey = OffloadKey
    abstract.PrepareStoreOutput = PrepareStoreOutput
    abstract.OffloadingEvent = OffloadingEvent


def import_native_manager(monkeypatch: pytest.MonkeyPatch, with_native: bool = True):
    install_vllm_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "certus_connector.mediums", raising=False)
    monkeypatch.delitem(sys.modules, "certus_connector.native_manager", raising=False)
    if with_native:
        monkeypatch.setitem(sys.modules, "certus_native", types.ModuleType("certus_native"))
    else:
        monkeypatch.setitem(sys.modules, "certus_native", None)
    importlib.invalidate_caches()
    module = importlib.import_module("certus_connector.native_manager")
    return importlib.reload(module)


@pytest.fixture
def native_manager_module(monkeypatch: pytest.MonkeyPatch):
    return import_native_manager(monkeypatch)


@pytest.fixture
def engine():
    return Mock()


@pytest.fixture
def manager(native_manager_module, engine):
    return native_manager_module.NativeCertusOffloadingManager(engine)


class TestNativeHelpers:
    def test_key_to_u64_accepts_int(self, native_manager_module):
        assert native_manager_module._key_to_u64(1234) == 1234

    def test_key_to_u64_uses_first_eight_bytes(self, native_manager_module):
        key = make_key(0x0102030405060708, suffix=0xA0B0C0D0)
        assert native_manager_module._key_to_u64(key) == 0x0102030405060708

    def test_key_to_u64_accepts_short_bytes(self, native_manager_module):
        assert native_manager_module._key_to_u64(b"abc") == int.from_bytes(b"abc", "big")

    def test_keys_to_u64s_converts_iterable(self, native_manager_module):
        keys = [make_key(1), 7, bytes.fromhex("0000000000000009") + b"tail"]
        assert native_manager_module._keys_to_u64s(keys) == [1, 7, 9]


class TestNativeCertusOffloadingManager:
    def test_lookup_hit_returns_true(self, manager, engine):
        engine.batch_check.return_value = 2
        assert manager.lookup(make_key(11)) is True
        engine.batch_check.assert_called_once_with([11])

    def test_lookup_miss_returns_false(self, manager, engine):
        engine.batch_check.return_value = 0
        assert manager.lookup(make_key(22)) is False
        engine.batch_check.assert_called_once_with([22])

    def test_prepare_load_returns_spec_with_locations(self, manager, engine, native_manager_module):
        keys = [make_key(1), make_key(2)]
        engine.prepare_load.return_value = [(101, 64), (202, 128)]

        spec = manager.prepare_load(keys)

        assert isinstance(spec, native_manager_module.CertusLoadStoreSpec)
        assert [(loc.nvme_slab, loc.dram_ptr, loc.size) for loc in spec.locations] == [
            (1, 101, 64),
            (2, 202, 128),
        ]
        engine.prepare_load.assert_called_once_with([1, 2])

    def test_prepare_load_accepts_generator(self, manager, engine):
        engine.prepare_load.return_value = [(1, 16), (2, 32)]

        spec = manager.prepare_load(make_key(i) for i in [7, 8])

        assert [loc.nvme_slab for loc in spec.locations] == [7, 8]
        engine.prepare_load.assert_called_once_with([7, 8])

    def test_touch_delegates_to_engine(self, manager, engine):
        manager.touch([make_key(3), make_key(4)])
        engine.touch.assert_called_once_with([3, 4])

    def test_complete_load_delegates_to_engine(self, manager, engine):
        manager.complete_load([make_key(5), make_key(6)])
        engine.complete_load.assert_called_once_with([5, 6])

    def test_prepare_store_returns_none_when_engine_rejects(self, manager, engine):
        engine.prepare_store.return_value = None

        assert manager.prepare_store([make_key(1), make_key(2)]) is None
        engine.prepare_store.assert_called_once_with([1, 2])
        assert list(manager.take_events()) == []

    def test_prepare_store_builds_output_and_eviction_event(self, manager, engine, native_manager_module):
        key_a = make_key(10, suffix=1)
        key_b = make_key(20, suffix=2)
        engine.prepare_store.return_value = ([20, 10], [90, 91])

        result = manager.prepare_store([key_a, key_b])

        assert result.keys_to_store == [key_b, key_a]
        assert isinstance(result.store_spec, native_manager_module.CertusLoadStoreSpec)
        assert [loc.nvme_slab for loc in result.store_spec.locations] == [20, 10]
        assert all(loc.dram_slot is None for loc in result.store_spec.locations)
        assert result.evicted_keys == [
            (90).to_bytes(8, "big"),
            (91).to_bytes(8, "big"),
        ]

        events = list(manager.take_events())
        assert len(events) == 1
        assert events[0].keys == result.evicted_keys
        assert events[0].medium == native_manager_module.CertusLoadStoreSpec.medium()
        assert events[0].removed is True
        assert list(manager.take_events()) == []

    def test_prepare_store_handles_empty_store_list(self, manager, engine, native_manager_module):
        engine.prepare_store.return_value = ([], [])

        result = manager.prepare_store([make_key(1)])

        assert result.keys_to_store == []
        assert result.evicted_keys == []
        assert isinstance(result.store_spec, native_manager_module.CertusLoadStoreSpec)
        assert result.store_spec.locations == []
        assert list(manager.take_events()) == []

    def test_complete_store_passes_success_true(self, manager, engine):
        manager.complete_store([make_key(30)], success=True)
        engine.complete_store.assert_called_once_with([30], True)

    def test_complete_store_passes_success_false(self, manager, engine):
        manager.complete_store([make_key(31), make_key(32)], success=False)
        engine.complete_store.assert_called_once_with([31, 32], False)

    def test_take_events_yields_then_clears(self, manager, native_manager_module):
        event = native_manager_module.OffloadingEvent(
            keys=[b"evicted"],
            medium=native_manager_module.CertusLoadStoreSpec.medium(),
            removed=True,
        )
        manager._events.append(event)

        assert list(manager.take_events()) == [event]
        assert list(manager.take_events()) == []

    def test_shutdown_delegates_to_engine(self, manager, engine):
        manager.shutdown()
        engine.shutdown.assert_called_once_with()

    def test_constructor_raises_importerror_without_certus_native(self, monkeypatch: pytest.MonkeyPatch):
        module = import_native_manager(monkeypatch, with_native=False)

        with pytest.raises(ImportError, match="certus_native is not installed"):
            module.NativeCertusOffloadingManager(Mock())
