# SPDX-License-Identifier: Apache-2.0
"""Unit tests for certus_connector/spec.py.

Runs without vllm, torch, SPDK, or certus_native hardware.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest

for mod_name in [
    "vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.kv_cache_utils",
    "vllm.v1.kv_offload", "vllm.v1.kv_offload.abstract",
    "vllm.v1.kv_offload.mediums", "vllm.v1.kv_offload.worker",
    "vllm.v1.kv_offload.worker.worker", "vllm.v1.kv_offload.spec",
    "vllm.v1.kv_cache_interface", "vllm.config", "vllm.logger",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

from abc import ABC, abstractmethod
from typing import NewType

OffloadKey = NewType("OffloadKey", bytes)

class LoadStoreSpec(ABC):
    @staticmethod
    @abstractmethod
    def medium() -> str: ...

class OffloadingManager(ABC):
    @abstractmethod
    def lookup(self, key, req_context=None): ...
    @abstractmethod
    def prepare_load(self, keys, req_context=None): ...
    @abstractmethod
    def prepare_store(self, keys, req_context=None): ...
    def touch(self, keys): pass
    def complete_load(self, keys): pass
    def complete_store(self, keys, success=True): pass
    def take_events(self): return ()
    def shutdown(self): pass

class OffloadingSpec:
    def __init__(self, vllm_config, kv_cache_config):
        self.extra_config = {}
        self.gpu_block_size = [131072]
        self.block_size_factor = 1

sys.modules["vllm.v1.kv_offload.abstract"].LoadStoreSpec = LoadStoreSpec
sys.modules["vllm.v1.kv_offload.abstract"].OffloadingManager = OffloadingManager
sys.modules["vllm.v1.kv_offload.abstract"].OffloadKey = OffloadKey
sys.modules["vllm.v1.kv_offload.spec"].OffloadingSpec = OffloadingSpec
sys.modules["vllm.logger"].init_logger = lambda name: __import__("logging").getLogger(name)

mock_engine = MagicMock()
mock_certus_native = types.ModuleType("certus_native")
mock_certus_native.CertusEngine = MagicMock(return_value=mock_engine)
sys.modules["certus_native"] = mock_certus_native

import certus_connector.spec as spec_module
from certus_connector.spec import _get_or_create_engine


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the process-level singleton before each test."""
    original = spec_module._ENGINE_SINGLETON
    spec_module._ENGINE_SINGLETON = None
    yield
    spec_module._ENGINE_SINGLETON = original
    mock_certus_native.CertusEngine.reset_mock()


class TestGetOrCreateEngine:
    def test_first_call_creates_engine(self):
        engine = _get_or_create_engine({"data_pci_addrs": ["0000:01:00.0"]})
        assert engine is not None
        mock_certus_native.CertusEngine.assert_called_once()

    def test_second_call_returns_same_instance(self):
        engine1 = _get_or_create_engine({})
        engine2 = _get_or_create_engine({"data_pci_addrs": ["different"]})
        assert engine1 is engine2
        assert mock_certus_native.CertusEngine.call_count == 1

    def test_config_passed_to_engine(self):
        _get_or_create_engine({
            "data_pci_addrs": ["0000:02:00.0"],
            "slab_size_bytes": 65536,
            "dram_cache_bytes": 1073741824,
        })
        call_kwargs = mock_certus_native.CertusEngine.call_args[0][0]
        assert call_kwargs["slab_size_bytes"] == 65536
        assert call_kwargs["dram_cache_bytes"] == 1073741824

    def test_reset_allows_new_engine(self):
        _get_or_create_engine({})
        spec_module._ENGINE_SINGLETON = None
        _get_or_create_engine({})
        assert mock_certus_native.CertusEngine.call_count == 2
