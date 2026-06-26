# SPDX-License-Identifier: Apache-2.0
"""Unit tests for certus_connector/handler.py.

Runs without SPDK, CUDA, or vllm. All dependencies mocked.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

for mod_name in [
    "vllm", "vllm.v1", "vllm.v1.kv_offload", "vllm.v1.kv_offload.abstract",
    "vllm.v1.kv_offload.mediums", "vllm.v1.kv_offload.worker",
    "vllm.v1.kv_offload.worker.worker", "vllm.logger",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

from dataclasses import dataclass

@dataclass
class TransferResult:
    job_id: int
    success: bool
    transfer_size: int
    transfer_time: float
    transfer_type: object

@dataclass
class GPULoadStoreSpec:
    block_ids: list

sys.modules["vllm.v1.kv_offload.mediums"].GPULoadStoreSpec = GPULoadStoreSpec
sys.modules["vllm.v1.kv_offload.worker.worker"].OffloadingHandler = object
sys.modules["vllm.v1.kv_offload.worker.worker"].TransferResult = TransferResult
sys.modules["vllm.v1.kv_offload.worker.worker"].TransferSpec = object
sys.modules["vllm.v1.kv_offload.worker.worker"].TransferType = object
sys.modules["vllm.logger"].init_logger = lambda n: __import__("logging").getLogger(n)

for mod_name in ["certus_native"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

from certus_connector.handler import CompletionDispatcher
from certus_connector.mediums import BlockLocation, CertusLoadStoreSpec


class MockEngine:
    def __init__(self):
        self._completions = []

    def poll_completions(self):
        result = list(self._completions)
        self._completions.clear()
        return result

    def store_async(self, job_id, gpu_block_ids, keys): pass
    def load_dma(self, job_id, gpu_block_ids, src_ptrs): pass
    def wait_job(self, job_id): pass


class TestCompletionDispatcher:
    @pytest.fixture
    def engine(self):
        return MockEngine()

    @pytest.fixture
    def dispatcher(self, engine):
        return CompletionDispatcher(engine)

    def test_store_completion_routed_to_store_buf(self, dispatcher, engine):
        dispatcher.register_store(1)
        engine._completions = [(1, True)]
        result = dispatcher.poll_stores()
        assert result == {1: True}
        assert dispatcher.poll_loads() == {}

    def test_load_completion_routed_to_load_buf(self, dispatcher, engine):
        dispatcher.register_load(2)
        engine._completions = [(2, True)]
        result = dispatcher.poll_loads()
        assert result == {2: True}
        assert dispatcher.poll_stores() == {}

    def test_unregistered_job_silently_dropped(self, dispatcher, engine):
        """Bug documented: job IDs not in store_jobs or load_jobs are silently dropped."""
        engine._completions = [(999, True)]
        stores = dispatcher.poll_stores()
        loads = dispatcher.poll_loads()
        assert 999 not in stores
        assert 999 not in loads

    def test_store_and_load_completions_in_same_poll(self, dispatcher, engine):
        dispatcher.register_store(10)
        dispatcher.register_load(20)
        engine._completions = [(10, True), (20, False)]
        stores = dispatcher.poll_stores()
        loads = dispatcher.poll_loads()
        assert stores.get(10) is True
        assert loads.get(20) is False

    def test_poll_stores_clears_buffer(self, dispatcher, engine):
        dispatcher.register_store(5)
        engine._completions = [(5, True)]
        dispatcher.poll_stores()
        assert dispatcher.poll_stores() == {}

    def test_poll_loads_clears_buffer(self, dispatcher, engine):
        dispatcher.register_load(6)
        engine._completions = [(6, False)]
        dispatcher.poll_loads()
        assert dispatcher.poll_loads() == {}

    def test_failed_store_completion_recorded(self, dispatcher, engine):
        dispatcher.register_store(7)
        engine._completions = [(7, False)]
        result = dispatcher.poll_stores()
        assert result == {7: False}

    def test_multiple_stores_all_routed(self, dispatcher, engine):
        for i in range(5):
            dispatcher.register_store(i)
        engine._completions = [(i, True) for i in range(5)]
        result = dispatcher.poll_stores()
        assert len(result) == 5
        assert all(v is True for v in result.values())
