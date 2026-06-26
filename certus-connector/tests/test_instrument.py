# SPDX-License-Identifier: Apache-2.0
"""Unit tests for certus_connector/_instrument.py.

Runs without vllm, torch, or certus_native.
"""
import sys
import threading
import types
from unittest.mock import MagicMock

import pytest

for mod_name in [
    "vllm", "vllm.v1", "vllm.v1.kv_offload", "vllm.v1.kv_offload.abstract",
    "vllm.v1.kv_offload.mediums", "vllm.v1.kv_offload.worker",
    "vllm.v1.kv_offload.worker.worker",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

import certus_connector._instrument as instr
from certus_connector._instrument import _percentile, Counters, start_reporter


class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 50) == 0.0

    def test_single_element_any_percentile(self):
        assert _percentile([42.0], 0) == 42.0
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 100) == 42.0

    def test_two_elements_p0(self):
        assert _percentile([2.0, 1.0], 0) == 1.0

    def test_two_elements_p100(self):
        assert _percentile([2.0, 1.0], 100) == 2.0

    def test_unsorted_input(self):
        data = [3.0, 1.0, 4.0, 1.5, 2.0]
        assert _percentile(data, 0) == 1.0

    def test_p50_median(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert _percentile(data, 50) == 30.0

    def test_p95(self):
        data = list(range(1, 101))
        result = _percentile([float(x) for x in data], 95)
        assert result == 96.0

    def test_duplicate_values(self):
        assert _percentile([5.0, 5.0, 5.0], 50) == 5.0


class TestCounters:
    def test_initial_values_are_zero(self):
        c = Counters()
        assert c.store_blocks_submitted == 0
        assert c.store_blocks_completed == 0
        assert c.store_total_bytes == 0
        assert c.load_blocks_submitted == 0
        assert c.load_blocks_completed == 0
        assert c.load_total_bytes == 0
        assert c.lookup_calls == 0
        assert c.evictions == 0

    def test_latency_lists_are_empty(self):
        c = Counters()
        assert c.store_latencies == []
        assert c.load_latencies == []


class TestStartReporter:
    def test_idempotent_no_double_thread(self, monkeypatch):
        """Calling start_reporter() twice must not spawn a second thread."""
        spawned = []

        class MockThread:
            def __init__(self, **kwargs):
                spawned.append(kwargs)
                self._mock = MagicMock()

            def start(self):
                pass

        monkeypatch.setattr(instr, "_reporter_started", False)
        monkeypatch.setattr(threading, "Thread", MockThread)

        start_reporter()
        start_reporter()

        assert len(spawned) == 1, "start_reporter must be idempotent — only one thread spawned"
