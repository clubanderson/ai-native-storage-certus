# SPDX-License-Identifier: Apache-2.0
"""Unit tests for regression_check.py — stdlib + pytest only, no hardware required."""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent))

import regression_check  # noqa: E402
from regression_check import calibrate, check_regression, run_replay  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────

def make_results(throughput_mbps: float = 200.0, hit_ratio: float = 0.95) -> dict:
    return {
        "handler": {"throughput_mbps": throughput_mbps},
        "manager": {"lookup": {"hit_ratio": hit_ratio}},
    }


def make_baseline(throughput_mbps: float = 200.0, tolerance_pct: int = 15) -> dict:
    return {"throughput_mbps": throughput_mbps, "tolerance_pct": tolerance_pct}


# ── check_regression() ────────────────────────────────────────────────────

class TestCheckRegression:
    def test_pass_at_exact_lower_bound(self):
        baselines = {"cpu": make_baseline(200.0, 15)}
        assert check_regression("cpu", make_results(170.0), baselines) is True

    def test_pass_above_baseline(self):
        baselines = {"cpu": make_baseline(200.0, 15)}
        assert check_regression("cpu", make_results(250.0), baselines) is True

    def test_fail_below_lower_bound(self):
        baselines = {"cpu": make_baseline(200.0, 15)}
        assert check_regression("cpu", make_results(169.9), baselines) is False

    def test_missing_connector_returns_false(self):
        baselines = {}
        assert check_regression("cpu", make_results(200.0), baselines) is False

    def test_zero_tolerance_pass_at_exact_baseline(self):
        baselines = {"cpu": make_baseline(200.0, 0)}
        assert check_regression("cpu", make_results(200.0), baselines) is True

    def test_zero_tolerance_fail_one_tenth_below(self):
        baselines = {"cpu": make_baseline(200.0, 0)}
        assert check_regression("cpu", make_results(199.9), baselines) is False

    def test_100_pct_tolerance_always_passes(self):
        baselines = {"cpu": make_baseline(200.0, 100)}
        assert check_regression("cpu", make_results(0.0), baselines) is True

    def test_prints_pass_status(self, capsys):
        baselines = {"cpu": make_baseline(200.0, 15)}
        check_regression("cpu", make_results(200.0), baselines)
        assert "PASS" in capsys.readouterr().out

    def test_prints_fail_status_and_deficit(self, capsys):
        baselines = {"cpu": make_baseline(200.0, 15)}
        check_regression("cpu", make_results(100.0), baselines)
        captured = capsys.readouterr().out
        assert "FAIL" in captured
        assert "regression" in captured

    def test_certus_connector(self):
        baselines = {"certus": make_baseline(305.0, 15)}
        assert check_regression("certus", make_results(305.0), baselines) is True
        assert check_regression("certus", make_results(259.2), baselines) is True
        assert check_regression("certus", make_results(259.1), baselines) is False

    def test_return_type_is_bool(self):
        baselines = {"cpu": make_baseline(200.0, 15)}
        result = check_regression("cpu", make_results(200.0), baselines)
        assert isinstance(result, bool)


# ── calibrate() ───────────────────────────────────────────────────────────

class TestCalibrate:
    def test_saves_correct_json_structure(self, tmp_path):
        path = tmp_path / "baselines.json"
        calibrate("cpu", make_results(150.0, 0.88), {}, path)
        data = json.loads(path.read_text())
        assert data["cpu"]["throughput_mbps"] == pytest.approx(150.0)
        assert data["cpu"]["hit_ratio"] == pytest.approx(0.88)
        assert data["cpu"]["tolerance_pct"] == 15

    def test_returns_true_when_hardware_within_threshold(self, tmp_path):
        ref = {"cpu": {"throughput_mbps": 200.0}}
        ok = calibrate("cpu", make_results(200.0), ref, tmp_path / "b.json")
        assert ok is True

    def test_returns_false_when_hardware_30_pct_slower(self, tmp_path):
        ref = {"cpu": {"throughput_mbps": 200.0}}
        ok = calibrate("cpu", make_results(140.0), ref, tmp_path / "b.json")
        assert ok is False

    def test_returns_true_at_exactly_25_pct_deficit(self, tmp_path):
        ref = {"cpu": {"throughput_mbps": 200.0}}
        ok = calibrate("cpu", make_results(150.0), ref, tmp_path / "b.json")
        assert ok is True

    def test_warning_printed_on_slow_hardware(self, tmp_path, capsys):
        ref = {"cpu": {"throughput_mbps": 200.0}}
        calibrate("cpu", make_results(100.0), ref, tmp_path / "b.json")
        assert "WARNING" in capsys.readouterr().out

    def test_no_warning_on_normal_hardware(self, tmp_path, capsys):
        ref = {"cpu": {"throughput_mbps": 200.0}}
        calibrate("cpu", make_results(200.0), ref, tmp_path / "b.json")
        assert "WARNING" not in capsys.readouterr().out

    def test_no_warning_when_no_ref_baseline(self, tmp_path, capsys):
        calibrate("cpu", make_results(50.0), {}, tmp_path / "b.json")
        assert "WARNING" not in capsys.readouterr().out

    def test_preserves_existing_connectors(self, tmp_path):
        path = tmp_path / "b.json"
        path.write_text(json.dumps({"fs": make_baseline(5000.0, 10)}))
        calibrate("cpu", make_results(200.0), {}, path)
        data = json.loads(path.read_text())
        assert "fs" in data
        assert "cpu" in data

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "deep" / "sub" / "baselines.json"
        calibrate("cpu", make_results(), {}, nested)
        assert nested.exists()

    def test_overwrites_existing_connector(self, tmp_path):
        path = tmp_path / "b.json"
        path.write_text(json.dumps({"cpu": make_baseline(100.0, 10)}))
        calibrate("cpu", make_results(300.0), {}, path)
        data = json.loads(path.read_text())
        assert data["cpu"]["throughput_mbps"] == pytest.approx(300.0)


# ── run_replay() ────────────────────────────────────────────────────────────

class TestRunReplay:
    """Tests that mock subprocess.run and intercept tempfile output."""

    def _patch_run_with_output(self, tmp_path, payload: dict):
        out_file = tmp_path / "replay_out.json"
        out_file.write_text(json.dumps(payload))

        class FakeTempFile:
            name = str(out_file)
            def __enter__(self): return self
            def __exit__(self, *a): pass

        return (
            patch("regression_check.subprocess.run", return_value=MagicMock(returncode=0)),
            patch("regression_check.tempfile.NamedTemporaryFile", return_value=FakeTempFile()),
        )

    def test_returns_parsed_results(self, tmp_path):
        payload = make_results(250.0)
        p1, p2 = self._patch_run_with_output(tmp_path, payload)
        with p1, p2:
            result = run_replay("cpu", "trace.jsonl", 32768)
        assert result["handler"]["throughput_mbps"] == pytest.approx(250.0)

    def test_returns_none_when_output_file_missing(self, tmp_path):
        missing = tmp_path / "no_such_file.json"

        class FakeTempFile:
            name = str(missing)
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch("regression_check.subprocess.run"), \
             patch("regression_check.tempfile.NamedTemporaryFile", return_value=FakeTempFile()):
            result = run_replay("cpu", "trace.jsonl", 32768)
        assert result is None

    def test_returns_none_on_malformed_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json {{{{")

        class FakeTempFile:
            name = str(bad)
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch("regression_check.subprocess.run"), \
             patch("regression_check.tempfile.NamedTemporaryFile", return_value=FakeTempFile()):
            result = run_replay("cpu", "trace.jsonl", 32768)
        assert result is None

    def test_subprocess_called_with_correct_connector(self, tmp_path):
        payload = make_results()
        p1, p2 = self._patch_run_with_output(tmp_path, payload)
        with p1 as mock_run, p2:
            run_replay("certus", "mytrace.jsonl", 8192)
        cmd = mock_run.call_args[0][0]
        assert "--connector" in cmd
        assert "certus" in cmd
        assert "--num-blocks" in cmd
        assert "8192" in cmd

    def test_subprocess_exit_code_not_checked(self, tmp_path):
        """Document Bug: subprocess.run has no check=True.

        Non-zero returncode does NOT cause run_replay to return None.
        This documents current (buggy) behavior.
        """
        payload = make_results()
        out_file = tmp_path / "out.json"
        out_file.write_text(json.dumps(payload))

        class FakeTempFile:
            name = str(out_file)
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch("regression_check.subprocess.run",
                   return_value=MagicMock(returncode=1)), \
             patch("regression_check.tempfile.NamedTemporaryFile", return_value=FakeTempFile()):
            result = run_replay("cpu", "trace.jsonl", 32768)
        assert result is not None, (
            "BUG: run_replay ignores non-zero subprocess returncode. "
            "Fix: add 'if proc.returncode != 0: return None' in run_replay()."
        )


# ── baselines.json schema validation ─────────────────────────────────────────

class TestReferencedBaselinesSchema:
    BASELINES_PATH = Path(__file__).parent / "baselines.json"

    @pytest.mark.skipif(
        not (Path(__file__).parent / "baselines.json").exists(),
        reason="baselines.json not yet present in this branch",
    )
    def test_baselines_json_has_required_connectors(self):
        data = json.loads(self.BASELINES_PATH.read_text())
        for connector in ("cpu", "fs", "certus"):
            assert connector in data, f"Missing connector '{connector}' in baselines.json"

    @pytest.mark.skipif(
        not (Path(__file__).parent / "baselines.json").exists(),
        reason="baselines.json not yet present in this branch",
    )
    def test_baselines_json_values_are_positive(self):
        data = json.loads(self.BASELINES_PATH.read_text())
        for connector, entry in data.items():
            assert entry["throughput_mbps"] > 0, f"{connector}: throughput must be positive"
            assert 0 < entry["tolerance_pct"] < 100, f"{connector}: tolerance_pct out of range"
