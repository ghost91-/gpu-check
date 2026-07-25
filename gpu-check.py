#!/usr/bin/env python3
"""GPU pre-purchase health check tool.

Runs a structured sequence of diagnostic checks on an NVIDIA GPU and writes
a pass/fail protocol for each step.  Designed for verifying used GPUs before
buying, but also useful for general health checking.

Usage:
    gpu-check.py --config configs/rtx-pro-6000-maxq.toml
    gpu-check.py --config configs/rtx-4070-laptop.toml --continue-on-fail
    gpu-check.py --config configs/rtx-5090.toml --skip-llm
    gpu-check.py --list-checks

All output is written to a timestamped protocol file in protocols/ and also
echoed to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

MIN_LOAD_POWER_W = 30
POWER_THRESHOLD_PCT = 0.85
VRAM_FILL_THRESHOLD_PCT = 0.80
MONITOR_BUFFER_S = 60
POWER_REREAD_DELAY_S = 0.1
LLAMA_SERVER_STARTUP_WAIT_S = 5
LLAMA_SERVER_READY_TIMEOUT_S = 120
AER_CORRECTABLE_WARN_THRESHOLD = 10

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    step: str
    name: str
    status: str  # PASS, FAIL, WARN, SKIP, MANUAL
    details: str
    raw_data: dict[str, Any] = field(default_factory=dict)
    action: str = ""  # explicit action text for WARN results


@dataclass
class GPUConfig:
    name: str
    expected_gpu_name: str
    expected_vram_mib: int
    expected_device_id: str
    expected_subsystem_vendor: str
    vbios_prefix: str
    expected_power_default_w: float
    expected_power_max_w: float
    expected_pcie_gen: int
    expected_pcie_width: int
    ecc_supported: bool
    ecc_expected_enabled: bool
    gpu_temp_max_c: float
    gpu_temp_walkaway_c: float
    mem_temp_max_c: float
    mem_temp_walkaway_c: float
    power_sustain_min_w: float
    idle_temp_max_c: float
    idle_power_max_w: float
    cooldown_time_s: int
    cooldown_temp_max_c: float
    gpuburn_duration_s: int
    gpufryer_duration_s: int
    expected_sm_clock_min_mhz: float
    expected_mem_clock_min_mhz: float
    smoke_model_file: str
    smoke_model_name: str
    smoke_expected_vram_mib: int
    smoke_context_length: int
    smoke_perf_baseline_label: str
    smoke_perf_tolerance_pct: float
    fill_model_file: str
    fill_model_name: str
    fill_expected_vram_mib: int
    fill_context_length: int
    models_dir: str
    gpu_burn_path: str
    gpu_fryer_path: str
    llama_server_path: str
    llama_bench_path: str
    llama_server_host: str
    llama_server_port: int
    smoke_prompt: str
    fill_prompt: str


# ---------------------------------------------------------------------------
# Protocol writer
# ---------------------------------------------------------------------------


class Protocol:
    def __init__(self, path: Path):
        self.path = path
        self.results: list[CheckResult] = []
        self._fh = open(path, "w", encoding="utf-8")
        self._write_header()

    def _write_header(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._fh.write("# GPU Health Check Protocol\n\n")
        self._fh.write(f"**Date:** {ts}\n\n")
        self._fh.write("---\n\n")
        self._fh.flush()

    def log(self, result: CheckResult):
        self.results.append(result)
        status_icon = {
            "PASS": "[PASS]",
            "FAIL": "[FAIL]",
            "WARN": "[WARN]",
            "SKIP": "[SKIP]",
            "MANUAL": "[MANUAL]",
        }.get(result.status, "[????]")
        line = f"{status_icon} {result.step}: {result.name}"
        if result.details:
            line += f" -- {result.details}"
        if result.action:
            line += f" | ACTION: {result.action}"
        self._fh.write(line + "\n")
        self._fh.flush()
        print(f"  {status_icon} {result.name}")
        if result.details:
            print(f"         {result.details}")
        if result.action:
            print(f"         >>> {result.action}")

    def log_info(self, msg: str):
        self._fh.write(f"\n{msg}\n\n")
        self._fh.flush()
        print(f"\n  {msg}")

    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        warned = sum(1 for r in self.results if r.status == "WARN")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        manual = sum(1 for r in self.results if r.status == "MANUAL")
        total = len(self.results)
        s = (
            f"\n--- Summary ---\n"
            f"Total checks: {total}\n"
            f"  PASS:    {passed}\n"
            f"  FAIL:    {failed}\n"
            f"  WARN:    {warned}\n"
            f"  SKIP:    {skipped}\n"
            f"  MANUAL:  {manual}\n"
        )
        if failed > 0:
            s += f"\n*** {failed} CHECK(S) FAILED ***\n"
            s += "\n*** DO NOT BUY *** Walk away. There is no return window.\n"
        elif warned > 0:
            s += f"\n*** {warned} WARNING(S) REQUIRE YOUR DECISION ***\n"
            s += "\nReview each warning below before deciding:\n"
            for r in self.results:
                if r.status == "WARN":
                    s += f"\n  [{r.step}] {r.name}\n"
                    if r.details:
                        s += f"    Details: {r.details}\n"
                    if r.action:
                        s += f"    ACTION: {r.action}\n"
                    else:
                        s += "    ACTION: Investigate manually before deciding.\n"
        elif manual > 0:
            s += f"\n*** {manual} manual check(s) require your decision ***\n"
        else:
            s += "\n*** All automated checks passed ***\n"
        self._fh.write(s)
        self._fh.flush()
        print(s)
        return s

    def close(self):
        self._fh.close()


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------


def run(cmd: list[str] | str, timeout: int = 30) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr).

    If cmd is a list, it is passed directly to subprocess (no shell).
    If cmd is a string, it is run via shell=True.
    """
    if isinstance(cmd, str):
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
        except subprocess.TimeoutExpired:
            return 124, "", f"Command timed out after {timeout}s"
        except FileNotFoundError as e:
            return 127, "", str(e)
    else:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
        except subprocess.TimeoutExpired:
            return 124, "", f"Command timed out after {timeout}s"
        except FileNotFoundError as e:
            return 127, "", str(e)


def run_csv(query: list[str], nounits: bool = True, field_names: list[str] | None = None) -> list[dict[str, str]]:
    """Run nvidia-smi with --format=csv,noheader and parse into list of dicts.

    Since nvidia-smi with 'noheader' produces no header row, we need field_names
    to map columns. If field_names is not provided, we extract them from the
    query argument (e.g. '--query-gpu=name,memory.total' -> ['name', 'memory.total']).
    """
    fmt = "csv,noheader"
    if nounits:
        fmt += ",nounits"
    cmd = ["nvidia-smi"] + query + [f"--format={fmt}"]
    rc, out, _ = run(cmd)
    if rc != 0 or not out:
        return []
    lines = [l for l in out.splitlines() if l.strip()]
    if not lines:
        return []
    ERROR_MARKERS = ("No devices", "No GPUs were found", "Failed to initialize",
                     "Driver library", "Permission denied", "is not a valid field",
                     "is not recognized")
    if any(any(marker in l for marker in ERROR_MARKERS) for l in lines):
        return []

    # Extract field names from the query argument
    if field_names is None:
        for arg in query:
            if arg.startswith("--query-gpu="):
                field_names = arg[len("--query-gpu=") :].split(",")
                break
            elif arg.startswith("--query-gpu"):
                # This shouldn't happen with our calling convention, but handle it
                pass
    if field_names is None:
        field_names = []

    rows: list[dict[str, str]] = []
    for line in lines:
        values = [v.strip() for v in line.split(",")]
        if field_names and len(values) == len(field_names):
            rows.append(dict(zip(field_names, values)))
        elif len(values) == 1:
            rows.append({"value": values[0]})
        else:
            rows.append({f"col{i}": v for i, v in enumerate(values)})
    return rows


def query_gpu(fields: list[str]) -> dict[str, str] | None:
    """Query a single GPU via nvidia-smi, return field->value dict.

    If any field is invalid, retries with only the valid fields.
    """
    field_str = ",".join(fields)
    rows = run_csv([f"--query-gpu={field_str}"], field_names=fields)
    if not rows:
        valid_fields = []
        for f in fields:
            rc, out, _ = run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader,nounits"])
            if rc == 0 and out and "is not a valid field" not in out and "is not recognized" not in out:
                valid_fields.append(f)
        if not valid_fields:
            return None
        rows = run_csv([f"--query-gpu={','.join(valid_fields)}"], field_names=valid_fields)
        if not rows:
            return None
    row = rows[0]
    return {k.strip(): (v.strip() if v else "") for k, v in row.items()}


def parse_pci_bus_id() -> str:
    """Get the PCI bus ID in the format 0000:XX:YY.Z for sysfs lookup."""
    row = query_gpu(["pci.bus_id"])
    if not row:
        return ""
    bus_id = row.get("pci.bus_id", "")
    m = re.match(r"^([0-9A-Fa-f]{4,8}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-9A-Fa-f])$", bus_id)
    if m:
        domain = m.group(1).zfill(4)[-4:]
        bus_id = f"{domain}:{m.group(2)}:{m.group(3)}.{m.group(4)}"
    return bus_id


def get_aer_counters(pci_bus_id: str) -> dict[str, str] | None:
    """Read PCIe AER counters from sysfs."""
    # pci_bus_id is already normalized to "0000:01:00.0" by parse_pci_bus_id
    sysfs_id = pci_bus_id

    base = Path("/sys/bus/pci/devices") / sysfs_id
    if not base.exists():
        return None

    counters: dict[str, str] = {}
    for fname in ["aer_dev_correctable", "aer_dev_nonfatal", "aer_dev_fatal"]:
        fpath = base / fname
        if fpath.exists():
            counters[fname] = fpath.read_text().strip()
        else:
            counters[fname] = "NOT_AVAILABLE"
    return counters


def parse_aer_total(aer_text: str) -> int:
    """Extract the TOTAL_ERR_xx number from an AER counter file."""
    if not aer_text or aer_text == "NOT_AVAILABLE":
        return 0
    for line in aer_text.splitlines():
        if line.startswith("TOTAL_ERR"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
    return 0


def parse_aer_breakdown(aer_text: str) -> dict[str, int]:
    """Parse all named counters from an AER file."""
    result: dict[str, int] = {}
    if not aer_text or aer_text == "NOT_AVAILABLE":
        return result
    for line in aer_text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] != "TOTAL_ERR":
            try:
                result[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return result


def kernel_log_grep(pattern: str, since_minutes: int = 0) -> list[str] | None:
    """Search kernel logs for a pattern.

    Returns None when no log source is available (both journalctl and dmesg
    return empty/error), so callers can distinguish "no errors found" from
    "could not read logs".
    """
    if since_minutes > 0:
        cmd = f'journalctl -k --since "{since_minutes} min ago" --no-pager 2>/dev/null'
    else:
        cmd = "journalctl -k --no-pager 2>/dev/null"
    rc, out, _ = run(cmd, timeout=15)
    matches = []
    journal_available = rc == 0 and bool(out)
    if journal_available:
        for line in out.splitlines():
            if re.search(pattern, line, re.IGNORECASE):
                matches.append(line)
    if not matches:
        rc2, out2, _ = run("dmesg 2>/dev/null", timeout=15)
        dmesg_available = rc2 == 0 and bool(out2)
        if dmesg_available:
            for line in out2.splitlines():
                if re.search(pattern, line, re.IGNORECASE):
                    matches.append(line)
    if not journal_available and not dmesg_available:
        return None
    return matches


def get_pcie_link_info() -> dict[str, str]:
    """Get PCIe link info from nvidia-smi -q output."""
    rc, out, _ = run("nvidia-smi -q", timeout=15)
    info: dict[str, str] = {}
    if rc != 0 or not out:
        return info
    lines = out.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "PCIe Generation":
            header_indent = len(line) - len(line.lstrip())
            j = i + 1
            while j < len(lines):
                sub_line = lines[j]
                sub_indent = len(sub_line) - len(sub_line.lstrip()) if sub_line.strip() else header_indent + 1
                if sub_indent <= header_indent:
                    break
                sub = sub_line.strip()
                parts = sub.split(":", 1)
                val = parts[1].strip() if len(parts) > 1 else ""
                if sub.startswith("Max") and val:
                    info["pcie_gen_max"] = val
                elif sub.startswith("Current") and val and "pcie_gen_current" not in info:
                    info["pcie_gen_current"] = val
                j += 1
        elif stripped == "Link Width":
            header_indent = len(line) - len(line.lstrip())
            j = i + 1
            while j < len(lines):
                sub_line = lines[j]
                sub_indent = len(sub_line) - len(sub_line.lstrip()) if sub_line.strip() else header_indent + 1
                if sub_indent <= header_indent:
                    break
                sub = sub_line.strip()
                parts = sub.split(":", 1)
                val = parts[1].strip() if len(parts) > 1 else ""
                if sub.startswith("Max") and val:
                    info["pcie_width_max"] = val
                elif sub.startswith("Current") and val and "pcie_width_current" not in info:
                    info["pcie_width_current"] = val
                j += 1
        elif "Replays Since Reset" in stripped:
            parts = stripped.split(":", 1)
            if len(parts) > 1:
                info["pcie_replays"] = parts[1].strip()
        elif "Replay Number Rollovers" in stripped:
            parts = stripped.split(":", 1)
            if len(parts) > 1:
                info["pcie_replay_rollovers"] = parts[1].strip()
    return info


def get_ecc_summary() -> dict[str, str]:
    """Parse ECC error counts from nvidia-smi -q -d ECC."""
    rc, out, _ = run("nvidia-smi -q -d ECC", timeout=15)
    info: dict[str, str] = {}
    if rc != 0 or not out:
        return info
    lines = out.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "ECC Mode":
            for j in range(i + 1, min(i + 4, len(lines))):
                sub = lines[j].strip()
                if sub.startswith("Current"):
                    info["ecc_mode_current"] = sub.split(":")[1].strip()
                elif sub.startswith("Pending"):
                    info["ecc_mode_pending"] = sub.split(":")[1].strip()
        elif "Volatile" in stripped and "ECC Errors" not in stripped:
            pass
        elif stripped.startswith("Single Bit"):
            for j in range(i + 1, min(i + 4, len(lines))):
                sub = lines[j].strip()
                if sub.startswith("Aggregate"):
                    info["ecc_single_aggregate"] = sub.split(":")[1].strip()
        elif stripped.startswith("Double Bit"):
            for j in range(i + 1, min(i + 4, len(lines))):
                sub = lines[j].strip()
                if sub.startswith("Aggregate"):
                    info["ecc_double_aggregate"] = sub.split(":")[1].strip()
        elif stripped.startswith("DRAM"):
            for j in range(i + 1, min(i + 4, len(lines))):
                sub = lines[j].strip()
                if "Correctable" in sub:
                    parts = sub.split(":", 1)
                    if len(parts) > 1:
                        info["ecc_dram_correctable"] = parts[1].strip()
                elif "Uncorrectable" in sub:
                    parts = sub.split(":", 1)
                    if len(parts) > 1:
                        info["ecc_dram_uncorrectable"] = parts[1].strip()
    return info


def get_row_remapper() -> dict[str, str]:
    """Parse row remapper status from nvidia-smi -q -d ROW_REMAPPER."""
    rc, out, _ = run("nvidia-smi -q -d ROW_REMAPPER", timeout=15)
    info: dict[str, str] = {}
    if rc != 0 or not out:
        return info
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Correctable Error"):
            info["remap_correctable"] = stripped.split(":")[1].strip()
        elif stripped.startswith("Uncorrectable Error"):
            info["remap_uncorrectable"] = stripped.split(":")[1].strip()
        elif stripped.startswith("Pending"):
            info["remap_pending"] = stripped.split(":")[1].strip()
        elif stripped.startswith("Remapping Failure"):
            info["remap_failure"] = stripped.split(":")[1].strip()
    return info


def get_retired_pages() -> dict[str, str]:
    """Parse retired pages from nvidia-smi -q -d PAGE_RETIREMENT."""
    rc, out, _ = run("nvidia-smi -q -d PAGE_RETIREMENT", timeout=15)
    info: dict[str, str] = {}
    if rc != 0 or not out:
        return info
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Single Bit ECC"):
            info["retired_single_bit"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Double Bit ECC"):
            info["retired_double_bit"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Pending"):
            info["retired_pending"] = stripped.split(":", 1)[1].strip()
    return info


def get_live_metrics() -> dict[str, str]:
    """Query current GPU metrics (temp, power, clocks, fan, memory)."""
    return (
        query_gpu(
            [
                "temperature.gpu",
                "temperature.memory",
                "power.draw",
                "clocks.sm",
                "clocks.mem",
                "fan.speed",
                "memory.used",
                "memory.free",
                "pstate",
            ]
        )
        or {}
    )


def get_tlimit_thresholds() -> dict[str, str]:
    """Parse T.Limit threshold deltas from nvidia-smi -q -d TEMPERATURE.

    These are thermal margin deltas (in degrees) before shutdown/slowdown/max-ops.
    Values near zero or negative indicate corrupted VBIOS thermal config.
    """
    rc, out, _ = run("nvidia-smi -q -d TEMPERATURE", timeout=15)
    info: dict[str, str] = {}
    if rc != 0 or not out:
        return info
    for line in out.splitlines():
        stripped = line.strip()
        if "T.Limit" in stripped and ":" in stripped:
            parts = stripped.split(":", 1)
            key = parts[0].strip()
            val = parts[1].strip() if len(parts) > 1 else ""
            info[key] = val
    return info


def parse_int(val: str) -> int:
    """Parse an integer from a possibly messy string."""
    if not val or val in ("N/A", "[N/A]", " ", "-"):
        return 0
    cleaned = re.sub(r"[^\d\-]", "", val)
    return int(cleaned) if cleaned else 0


def parse_float(val: str) -> float:
    """Parse a float from a possibly messy string."""
    if not val or val in ("N/A", "[N/A]", " ", "-"):
        return 0.0
    cleaned = re.sub(r"[^\d.\-]", "", val)
    return float(cleaned) if cleaned else 0.0


def parse_watts(val: str) -> float:
    """Parse wattage, handling N/A."""
    if not val or val.upper().startswith("N/A"):
        return -1.0
    return parse_float(val)


def read_power_draw(max_plausible_w: float = 0) -> float:
    """Read power.draw, discarding the first reading after GPU state transitions.

    nvidia-smi returns a garbage power value (590.01W) on the first query
    after the GPU transitions from P8 sleep to P0. This is a driver bug
    affecting both --query-gpu and -q -d POWER paths. The second query
    returns the correct value.

    If max_plausible_w > 0, re-reads once if the first value exceeds it.
    """
    val = parse_watts((query_gpu(["power.draw"]) or {}).get("power.draw", ""))
    if max_plausible_w <= 0 or (val >= 0 and val <= max_plausible_w):
        return val
    time.sleep(POWER_REREAD_DELAY_S)
    return parse_watts((query_gpu(["power.draw"]) or {}).get("power.draw", ""))


# ---------------------------------------------------------------------------
# Check definitions
# ---------------------------------------------------------------------------

CHECK_NAMES = [
    ("identity", "GPU identity and board info"),
    ("power-limits", "Power limit verification (detect shunt mods / VBIOS tampering)"),
    ("pcie-link", "PCIe link generation and width"),
    ("tlimit-thresholds", "Thermal T.Limit threshold sanity check"),
    ("ecc-baseline", "ECC status and error counts (baseline)"),
    ("row-remapper-baseline", "Row remapper status (baseline)"),
    ("retired-pages-baseline", "Retired pages status (baseline)"),
    ("aer-baseline", "PCIe AER counters (baseline)"),
    ("idle-thermals", "Idle thermals and power"),
    ("kernel-log-baseline", "Kernel log scan for Xid/AER/NVRM (baseline)"),
    ("stress-test-gpuburn", "gpu-burn stress test (ALU error detection, clock verification)"),
    ("pcie-under-load", "PCIe link under load (gpu-burn)"),
    ("clock-verification", "Clock verification under load"),
    ("stress-test-gpufryer", "gpu-fryer stress test (full TGP power/thermal, VRAM fill)"),
    ("power-verification", "Power draw verification under full TGP load"),
    ("vram-fill-verification", "VRAM fill verification under load"),
    ("pcie-under-load-gpufryer", "PCIe link under full TGP load (gpu-fryer)"),
    ("clock-verification-gpufryer", "Clock and throttle verification under full TGP (gpu-fryer)"),
    ("ecc-post", "ECC error counts (post-stress)"),
    ("row-remapper-post", "Row remapper status (post-stress)"),
    ("retired-pages-post", "Retired pages status (post-stress)"),
    ("aer-post", "PCIe AER counters (post-stress)"),
    ("aer-post-delta", "PCIe AER error delta (post-stress)"),
    ("kernel-log-post-stress", "Kernel log scan (post-stress)"),
    ("llm-smoke", "LLM smoke test (small model)"),
    ("llm-fill", "LLM VRAM-fill test (large model)"),
    ("llm-bench", "llama-bench performance benchmark"),
    ("ecc-final", "ECC error counts (final)"),
    ("row-remapper-final", "Row remapper status (final)"),
    ("retired-pages-final", "Retired pages status (final)"),
    ("aer-final", "PCIe AER counters (final)"),
    ("aer-final-delta", "PCIe AER error delta (final)"),
    ("kernel-log-final", "Kernel log scan (final)"),
    ("cooldown", "Post-test cooldown verification"),
    ("bug-report", "nvidia-bug-report capture"),
]


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------


class GPUChecker:
    def __init__(self, config: GPUConfig, protocol: Protocol, args: argparse.Namespace):
        self.cfg = config
        self.proto = protocol
        self.args = args
        self.pci_bus_id = ""
        self.baselines: dict[str, Any] = {}
        self.gpuburn_monitor = None
        self.gpufryer_monitor = None

    def _fail(self, step: str, name: str, details: str, data: dict | None = None):
        self.proto.log(
            CheckResult(step=step, name=name, status="FAIL", details=details, raw_data=data or {})
        )
        if not self.args.continue_on_fail:
            self.proto.log_info("Stopping due to failure (--continue-on-fail not set).")
            self._print_walkaway_reminder()
            sys.exit(1)

    def _pass(self, step: str, name: str, details: str, data: dict | None = None):
        self.proto.log(
            CheckResult(step=step, name=name, status="PASS", details=details, raw_data=data or {})
        )

    def _warn(self, step: str, name: str, details: str, data: dict | None = None, action: str = ""):
        self.proto.log(
            CheckResult(step=step, name=name, status="WARN", details=details, raw_data=data or {}, action=action)
        )

    def _skip(self, step: str, name: str, details: str):
        self.proto.log(
            CheckResult(step=step, name=name, status="SKIP", details=details, raw_data={})
        )

    def _manual(self, step: str, name: str, details: str):
        self.proto.log(
            CheckResult(step=step, name=name, status="MANUAL", details=details, raw_data={})
        )

    def _print_walkaway_reminder(self):
        msg = (
            "\n*** REMEMBER: If anything is even slightly fishy, walk away. ***\n"
            "    This is a conservative test. There is no return window.\n"
        )
        self.proto.log_info(msg)

    # ---- Individual checks ----

    def check_identity(self):
        step = "identity"
        name = "GPU identity and board info"
        info = query_gpu(
            [
                "name",
                "uuid",
                "serial",
                "vbios_version",
                "driver_version",
                "pci.device_id",
                "pci.sub_device_id",
                "memory.total",
            ]
        )
        if not info:
            self._fail(step, name, "nvidia-smi query failed -- is the driver loaded?")
            return

        # board_part_number is not a queryable field; get it from nvidia-smi -q
        rc, q_out, _ = run("nvidia-smi -q", timeout=15)
        board_pn = ""
        if rc == 0 and q_out:
            for line in q_out.splitlines():
                stripped = line.strip()
                if stripped.startswith("Board Part Number"):
                    board_pn = stripped.split(":", 1)[1].strip()
                    break
        if board_pn:
            info["board_part_number"] = board_pn

        gpu_name = info.get("name", "")
        vram = info.get("memory.total", "0")
        device_id = info.get("pci.device_id", "")
        sub_device_id = info.get("pci.sub_device_id", "")
        vbios = info.get("vbios_version", "")
        serial = info.get("serial", "")

        self.pci_bus_id = parse_pci_bus_id()

        problems = []
        if self.cfg.expected_gpu_name and self.cfg.expected_gpu_name not in gpu_name:
            problems.append(f"name mismatch: got '{gpu_name}', expected to contain '{self.cfg.expected_gpu_name}'")
        if parse_int(vram) != self.cfg.expected_vram_mib:
            problems.append(
                f"VRAM mismatch: got {vram} MiB, expected {self.cfg.expected_vram_mib} MiB"
            )
        # nvidia-smi returns device_id as "0xDDDDVVVV" (device + vendor combined)
        # e.g. "0x282010DE" means device=2820, vendor=10DE
        # Config has "10DE:2820" format (vendor:device)
        dev_id_raw = device_id.upper().replace("0X", "")
        expected_dev = self.cfg.expected_device_id.upper().replace("0X", "")
        if ":" in expected_dev:
            vendor_part, device_part = expected_dev.split(":")
            # Try both orderings: vendor+device and device+vendor
            expected_vd = vendor_part + device_part
            expected_dv = device_part + vendor_part
            if dev_id_raw and dev_id_raw != expected_vd and dev_id_raw != expected_dv:
                problems.append(
                    f"device ID mismatch: got '{device_id}', expected '{self.cfg.expected_device_id}'"
                )
        elif dev_id_raw and expected_dev not in dev_id_raw:
            problems.append(
                f"device ID mismatch: got '{device_id}', expected '{self.cfg.expected_device_id}'"
            )
        if self.cfg.expected_subsystem_vendor:
            sub_raw = sub_device_id.upper().replace("0X", "")
            sub_vendor = ""
            if len(sub_raw) >= 8:
                sub_vendor = sub_raw[:4]
            expected_sub = self.cfg.expected_subsystem_vendor.upper().replace("0X", "")
            if sub_vendor and sub_vendor != expected_sub and sub_raw[4:8] != expected_sub:
                problems.append(
                    f"subsystem vendor mismatch: got '{sub_vendor}', "
                    f"expected '{self.cfg.expected_subsystem_vendor}'"
                )
        if self.cfg.vbios_prefix and not vbios.upper().startswith(
            self.cfg.vbios_prefix.upper()
        ):
            problems.append(
                f"VBIOS version '{vbios}' does not start with expected prefix "
                f"'{self.cfg.vbios_prefix}'"
            )

        details_parts = [
            f"name={gpu_name}",
            f"VRAM={vram} MiB",
            f"device_id={device_id}",
            f"sub_device_id={sub_device_id}",
            f"VBIOS={vbios}",
            f"serial={serial}",
        ]
        details = "; ".join(details_parts)

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", info)
        else:
            self._pass(step, name, details, info)

    def check_power_limits(self):
        step = "power-limits"
        name = "Power limit verification"
        info = query_gpu(
            ["power.limit", "power.default_limit", "power.min_limit", "power.max_limit"]
        )
        if not info:
            self._fail(step, name, "Could not query power limits")
            return

        pwr_limit = parse_watts(info.get("power.limit", ""))
        pwr_default = parse_watts(info.get("power.default_limit", ""))
        pwr_min = parse_watts(info.get("power.min_limit", ""))
        pwr_max = parse_watts(info.get("power.max_limit", ""))

        details = (
            f"limit={pwr_limit:.0f}W, default={pwr_default:.0f}W, "
            f"min={pwr_min:.0f}W, max={pwr_max:.0f}W"
        )

        problems = []
        if pwr_default > 0 and abs(pwr_default - self.cfg.expected_power_default_w) > 1:
            problems.append(
                f"default power limit {pwr_default:.0f}W != expected "
                f"{self.cfg.expected_power_default_w}W (possible VBIOS tampering)"
            )
        if pwr_max > 0 and pwr_max > self.cfg.expected_power_max_w + 1:
            problems.append(
                f"max power limit {pwr_max:.0f}W > expected "
                f"{self.cfg.expected_power_max_w}W (possible shunt mod or flashed VBIOS)"
            )
        if pwr_limit > 0 and pwr_default > 0 and abs(pwr_limit - pwr_default) > 1:
            problems.append(
                f"power limit {pwr_limit:.0f}W != default {pwr_default:.0f}W "
                f"(limit has been changed from stock)"
            )

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", info)
        else:
            self._pass(step, name, details, info)

    def check_pcie_link(self):
        step = "pcie-link"
        name = "PCIe link generation and width"
        info = get_pcie_link_info()
        if not info:
            self._fail(step, name, "Could not retrieve PCIe link info")
            return

        gen_current = parse_int(info.get("pcie_gen_current", "0"))
        gen_max = parse_int(info.get("pcie_gen_max", "0"))
        width_current = parse_int(info.get("pcie_width_current", "0"))
        width_max = parse_int(info.get("pcie_width_max", "0"))
        replays = parse_int(info.get("pcie_replays", "0"))
        replay_rollovers = parse_int(info.get("pcie_replay_rollovers", "0"))

        details = (
            f"gen={gen_current}x{width_current} (max {gen_max}x{width_max}), "
            f"replays={replays}, rollovers={replay_rollovers}"
        )

        problems = []
        warnings = []
        if gen_max < self.cfg.expected_pcie_gen:
            problems.append(
                f"max PCIe gen {gen_max} < expected {self.cfg.expected_pcie_gen}"
            )
        if gen_current < self.cfg.expected_pcie_gen:
            warnings.append(
                f"current PCIe gen {gen_current} < expected {self.cfg.expected_pcie_gen} "
                f"(may be normal at idle; re-check under load)"
            )
        if width_current < self.cfg.expected_pcie_width:
            warnings.append(
                f"current PCIe width x{width_current} < expected "
                f"x{self.cfg.expected_pcie_width} (may be normal at idle; re-check under load)"
            )
        if width_max < self.cfg.expected_pcie_width:
            problems.append(
                f"max PCIe width x{width_max} < expected "
                f"x{self.cfg.expected_pcie_width} (hardware limit)"
            )
        if replays > 0:
            problems.append(f"PCIe replay counter is non-zero: {replays}")
        if replay_rollovers > 0:
            problems.append(f"PCIe replay rollovers non-zero: {replay_rollovers}")

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", info)
        else:
            note = ""
            if warnings:
                note = "; ".join(warnings) + " (will be re-checked under load)"
            self._pass(step, name, f"{details}{'; ' + note if note else ''}", info)

    def _check_ecc_state(self, step: str, name: str, label: str) -> dict[str, str]:
        info = get_ecc_summary()
        if not info:
            if self.cfg.ecc_supported:
                self._fail(step, name, f"Could not query ECC status ({label})")
            else:
                self._skip(step, name, f"ECC not supported on this card ({label})")
            return {}

        ecc_mode = info.get("ecc_mode_current", "")
        single_agg = info.get("ecc_single_aggregate", "0")
        double_agg = info.get("ecc_double_aggregate", "0")
        dram_cor = info.get("ecc_dram_correctable", "0")
        dram_unc = info.get("ecc_dram_uncorrectable", "0")

        details = (
            f"mode={ecc_mode}, single_bit_agg={single_agg}, double_bit_agg={double_agg}, "
            f"dram_correctable={dram_cor}, dram_uncorrectable={dram_unc}"
        )

        problems = []
        if self.cfg.ecc_expected_enabled and ecc_mode.upper() != "ENABLED":
            problems.append(f"ECC mode is {ecc_mode}, expected Enabled")
        if parse_int(single_agg) > 0:
            problems.append(f"single-bit ECC errors present: {single_agg}")
        if parse_int(double_agg) > 0:
            problems.append(f"double-bit ECC errors present: {double_agg}")
        if parse_int(dram_cor) > 0:
            problems.append(f"DRAM correctable errors present: {dram_cor}")
        if parse_int(dram_unc) > 0:
            problems.append(f"DRAM uncorrectable errors present: {dram_unc}")

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", info)
        else:
            self._pass(step, name, f"{label}: {details}", info)
        return info

    def _check_row_remapper(self, step: str, name: str, label: str):
        info = get_row_remapper()
        if not info:
            self._skip(step, name, f"Row remapper not queryable ({label})")
            return

        cor = info.get("remap_correctable", "0")
        unc = info.get("remap_uncorrectable", "0")
        pending = info.get("remap_pending", "No")
        failure = info.get("remap_failure", "No")

        details = (
            f"correctable={cor}, uncorrectable={unc}, pending={pending}, failure={failure}"
        )

        problems = []
        if parse_int(cor) > 0:
            problems.append(f"correctable remapped rows present: {cor}")
        if parse_int(unc) > 0:
            problems.append(f"uncorrectable remapped rows present: {unc}")
        if pending.upper() == "YES":
            problems.append("row remapping is pending")
        if failure.upper() == "YES":
            problems.append("remapping failure occurred")

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", info)
        else:
            self._pass(step, name, f"{label}: {details}", info)

    def _check_retired_pages(self, step: str, name: str, label: str):
        info = get_retired_pages()
        if not info:
            self._skip(step, name, f"Retired pages not queryable ({label})")
            return

        single = info.get("retired_single_bit", "0")
        double = info.get("retired_double_bit", "0")
        pending = info.get("retired_pending", "")

        details = f"single_bit={single}, double_bit={double}, pending={pending}"

        problems = []
        if parse_int(single) > 0:
            problems.append(f"single-bit retired pages present: {single}")
        if parse_int(double) > 0:
            problems.append(f"double-bit retired pages present: {double}")
        if pending and pending.upper() not in ("NO", "NONE", "0", "N/A", "[N/A]", ""):
            problems.append(f"pending page retirements: {pending}")

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", info)
        else:
            self._pass(step, name, f"{label}: {details}", info)

    def _check_aer(self, step: str, name: str, label: str, store_baseline: bool = False) -> dict[str, str] | None:
        if not self.pci_bus_id:
            self.pci_bus_id = parse_pci_bus_id()
        if not self.pci_bus_id:
            self._fail(step, name, f"Could not determine PCI bus ID ({label})")
            return None

        counters = get_aer_counters(self.pci_bus_id)
        if not counters:
            self._skip(step, name, f"AER sysfs entries not found ({label})")
            return None

        corr_text = counters.get("aer_dev_correctable", "NOT_AVAILABLE")
        nonfatal_text = counters.get("aer_dev_nonfatal", "NOT_AVAILABLE")
        fatal_text = counters.get("aer_dev_fatal", "NOT_AVAILABLE")

        corr_total = parse_aer_total(corr_text)
        nonfatal_total = parse_aer_total(nonfatal_text)
        fatal_total = parse_aer_total(fatal_text)

        corr_breakdown = parse_aer_breakdown(corr_text)
        details_parts = [f"correctable_total={corr_total}"]
        if corr_breakdown:
            notable = {k: v for k, v in corr_breakdown.items() if v > 0}
            if notable:
                details_parts.append(f"correctable_breakdown={notable}")
        details_parts.append(f"nonfatal_total={nonfatal_total}")
        details_parts.append(f"fatal_total={fatal_total}")
        details = f"{label}: " + ", ".join(details_parts)

        problems = []
        if fatal_total > 0:
            problems.append(f"fatal AER errors present: {fatal_total}")
        if nonfatal_total > 0:
            problems.append(f"non-fatal AER errors present: {nonfatal_total}")

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", counters)
        else:
            self._pass(step, name, details, counters)
        if counters and store_baseline:
            self.baselines["aer"] = counters
        return counters

    def _check_aer_with_delta(self, step: str, name: str, label: str,
                              delta_suffix: str, warn_action: str) -> None:
        counters = self._check_aer(step, name, label)
        if counters and "aer" in self.baselines:
            baseline = self.baselines["aer"]
            for key in ["aer_dev_correctable", "aer_dev_nonfatal", "aer_dev_fatal"]:
                base_total = parse_aer_total(baseline.get(key, ""))
                curr_total = parse_aer_total(counters.get(key, ""))
                delta = curr_total - base_total
                err_label = key.replace("aer_dev_", "")
                if delta < 0:
                    self._fail(
                        f"aer-{delta_suffix}-delta",
                        f"AER {err_label} {'final ' if delta_suffix == 'final' else ''}delta check",
                        f"AER {err_label} counter decreased (baseline={base_total}, current={curr_total}) - counter reset indicates a PCI link reset event",
                        {"baseline": base_total, "current": curr_total, "delta": delta},
                    )
                elif delta > 0:
                    if "nonfatal" in err_label or "fatal" in err_label:
                        self._fail(
                            f"aer-{delta_suffix}-delta",
                            f"AER {err_label} {'final ' if delta_suffix == 'final' else ''}delta check",
                            f"{delta} new {err_label} errors during {'entire test session' if delta_suffix == 'final' else 'stress test'}",
                            {"baseline": base_total, "current": curr_total, "delta": delta},
                        )
                    elif delta > AER_CORRECTABLE_WARN_THRESHOLD:
                        self._warn(
                            f"aer-{delta_suffix}-delta",
                            f"AER {err_label} {'final ' if delta_suffix == 'final' else ''}delta check",
                            f"{delta} new correctable errors during {'entire test session' if delta_suffix == 'final' else 'stress test'}",
                            {"baseline": base_total, "current": curr_total, "delta": delta},
                            action=warn_action,
                        )

    def check_idle_thermals(self):
        step = "idle-thermals"
        name = "Idle thermals and power"
        info = get_live_metrics()
        if not info:
            self._fail(step, name, "Could not query live metrics")
            return

        gpu_temp = parse_float(info.get("temperature.gpu", "0"))
        mem_temp = parse_float(info.get("temperature.memory", "0"))
        power = read_power_draw(self.cfg.expected_power_max_w)
        fan = info.get("fan.speed", "")
        pstate = info.get("pstate", "")

        details = (
            f"gpu_temp={gpu_temp:.0f}C, mem_temp={mem_temp:.0f}C, "
            f"power={power:.1f}W, fan={fan}%, pstate={pstate}"
        )

        problems = []
        if gpu_temp > self.cfg.idle_temp_max_c:
            problems.append(
                f"idle GPU temp {gpu_temp:.0f}C > {self.cfg.idle_temp_max_c}C"
            )
        if power > 0 and power > self.cfg.idle_power_max_w:
            problems.append(
                f"idle power {power:.1f}W > {self.cfg.idle_power_max_w}W "
                f"(card may be stuck in a performance state)"
            )

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", info)
        else:
            self._pass(step, name, details, info)

    def check_tlimit_thresholds(self):
        step = "tlimit-thresholds"
        name = "Thermal T.Limit threshold sanity check"
        info = get_tlimit_thresholds()
        if not info:
            self._skip(step, name, "T.Limit thresholds not available via nvidia-smi")
            return

        details_parts = []
        warnings = []
        for key, val in info.items():
            details_parts.append(f"{key}={val}")
            numeric = parse_int(val)
            if "Shutdown" in key and "Specification" in key and numeric == 0:
                warnings.append(f"{key}=0 (shutdown threshold at zero, possible VBIOS corruption)")
            elif "Slowdown" in key and "Specification" in key and numeric == 0:
                warnings.append(f"{key}=0 (slowdown threshold at zero, possible VBIOS corruption)")
        details = ", ".join(details_parts)

        if warnings:
            self._pass(
                step, name,
                f"{details} (NOTE: {'; '.join(warnings)}; clock-verification step screens for the associated bug)",
                info,
            )
        else:
            self._pass(step, name, details, info)

    def check_kernel_log(self, step: str, name: str, label: str):
        matches = kernel_log_grep(r"NVRM:.*Xid|NVRM:.*\b(?:error|fail|fault|uncorrectable)\b.*:|pcieport.*AER", since_minutes=0)
        if matches is None:
            self._warn(
                step,
                name,
                "Could not read kernel logs (journalctl and dmesg both unavailable)",
                action="Verify that journalctl is running or that you have permissions for dmesg. Cannot verify kernel log for Xid errors.",
            )
            return
        if matches:
            details = f"{len(matches)} NVIDIA Xid entries found ({label})"
            self._fail(step, name, f"{details}: {matches[:3]}", {"matches": matches[:10]})
        else:
            self._pass(step, name, f"No NVIDIA Xid errors in kernel log ({label})", {})

    # ---- Stress test ----

    def _monitor_stress(
        self,
        proc: subprocess.Popen,
        duration_s: int,
    ) -> dict[str, Any]:
        """Run a per-second monitoring loop while a stress process runs.

        Returns a dict with min/max clocks, temps, power, throttle reasons,
        sample count, max VRAM usage, and the process exit handling.
        """
        print(f"  >>> Note: throttle detection samples at ~1s intervals; transient events <1s may be missed")
        monitor: dict[str, Any] = {
            "max_gpu_temp": 0.0,
            "max_mem_temp": 0.0,
            "min_sm_clock": 999999.0,
            "max_sm_clock": 0.0,
            "min_mem_clock": 999999.0,
            "max_mem_clock": 0.0,
            "min_power": 999999.0,
            "max_power": 0.0,
            "max_vram_used_mib": 0,
            "throttle_activated": [],
            "samples": 0,
        }
        throttle_fields = [
            "clocks_event_reasons.hw_thermal_slowdown",
            "clocks_event_reasons.sw_power_cap",
            "clocks_event_reasons.hw_power_brake_slowdown",
            "clocks_event_reasons.sw_thermal_slowdown",
        ]
        monitor_fields = [
            "temperature.gpu",
            "temperature.memory",
            "power.draw",
            "clocks.current.sm",
            "clocks.current.memory",
            "memory.used",
        ] + throttle_fields

        elapsed = 0
        while elapsed < duration_s + MONITOR_BUFFER_S:
            if proc.poll() is not None:
                break
            snap = query_gpu(monitor_fields) or {}
            if snap:
                gpu_t = parse_float(snap.get("temperature.gpu", "0"))
                mem_t = parse_float(snap.get("temperature.memory", "0"))
                power = parse_watts(snap.get("power.draw", ""))
                if self.cfg.expected_power_max_w > 0 and power > self.cfg.expected_power_max_w + 100:
                    power = read_power_draw(self.cfg.expected_power_max_w)
                sm_clock = parse_float(snap.get("clocks.current.sm", "0"))
                mem_clock = parse_float(snap.get("clocks.current.memory", "0"))
                vram_used = parse_int(snap.get("memory.used", "0"))

                if gpu_t > 0:
                    monitor["max_gpu_temp"] = max(monitor["max_gpu_temp"], gpu_t)
                if mem_t > 0:
                    monitor["max_mem_temp"] = max(monitor["max_mem_temp"], mem_t)
                if vram_used > 0:
                    monitor["max_vram_used_mib"] = max(monitor["max_vram_used_mib"], vram_used)

                if power > MIN_LOAD_POWER_W and sm_clock > 0:
                    monitor["min_sm_clock"] = min(monitor["min_sm_clock"], sm_clock)
                    monitor["max_sm_clock"] = max(monitor["max_sm_clock"], sm_clock)
                    monitor["min_power"] = min(monitor["min_power"], power)
                    monitor["max_power"] = max(monitor["max_power"], power)
                if mem_clock > 0:
                    monitor["min_mem_clock"] = min(monitor["min_mem_clock"], mem_clock)
                    monitor["max_mem_clock"] = max(monitor["max_mem_clock"], mem_clock)

                for field in throttle_fields:
                    val = snap.get(field, "")
                    if val and val.upper() == "ACTIVE" and field not in monitor["throttle_activated"]:
                        monitor["throttle_activated"].append(field)

                monitor["samples"] += 1

            if (
                "pcie_link_info" not in monitor
                and elapsed >= duration_s * 0.75
                and proc.poll() is None
            ):
                monitor["pcie_link_info"] = get_pcie_link_info()

            if elapsed > 0 and elapsed % 60 == 0:
                print(f"  >>> {elapsed}s: gpu_temp={monitor['max_gpu_temp']:.0f}C, "
                      f"sm_clock={monitor['min_sm_clock']:.0f}-{monitor['max_sm_clock']:.0f}MHz, "
                      f"power={monitor['min_power']:.0f}-{monitor['max_power']:.0f}W, "
                      f"samples={monitor['samples']}")
            time.sleep(1)
            elapsed += 1

        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, 0)
        except ProcessLookupError:
            pass
        else:
            os.killpg(pgid, 15)

        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, 9)
            out, err = proc.communicate()

        monitor["stdout"] = (out or "").strip()
        monitor["stderr"] = (err or "").strip()
        monitor["returncode"] = proc.returncode

        if monitor["min_sm_clock"] == 999999.0:
            monitor["min_sm_clock"] = 0.0
        if monitor["min_power"] == 999999.0:
            monitor["min_power"] = 0.0
        if monitor["min_mem_clock"] == 999999.0:
            monitor["min_mem_clock"] = 0.0

        print(f"  >>> Monitoring complete: {monitor['samples']} samples over {elapsed}s")
        print(f"  >>> Max GPU temp: {monitor['max_gpu_temp']:.0f}C, "
              f"SM clock range: {monitor['min_sm_clock']:.0f}-{monitor['max_sm_clock']:.0f}MHz, "
              f"power range: {monitor['min_power']:.0f}-{monitor['max_power']:.0f}W")
        if monitor["throttle_activated"]:
            print(f"  >>> Throttle reasons activated: {', '.join(monitor['throttle_activated'])}")

        return monitor

    def _eval_stress_thermals(self, monitor: dict[str, Any]) -> list[str]:
        """Check thermal thresholds from a monitoring dict. Returns problem list."""
        problems = []
        max_gpu_temp = monitor.get("max_gpu_temp", 0)
        max_mem_temp = monitor.get("max_mem_temp", 0)

        if max_gpu_temp > self.cfg.gpu_temp_walkaway_c:
            problems.append(
                f"GPU temp {max_gpu_temp:.0f}C > walkaway threshold "
                f"{self.cfg.gpu_temp_walkaway_c}C"
            )
        if max_gpu_temp > self.cfg.gpu_temp_max_c:
            problems.append(
                f"GPU temp {max_gpu_temp:.0f}C > max threshold {self.cfg.gpu_temp_max_c}C"
            )
        if max_mem_temp > 0 and max_mem_temp > self.cfg.mem_temp_walkaway_c:
            problems.append(
                f"memory temp {max_mem_temp:.0f}C > walkaway threshold "
                f"{self.cfg.mem_temp_walkaway_c}C"
            )
        if max_mem_temp > 0 and max_mem_temp > self.cfg.mem_temp_max_c:
            problems.append(
                f"memory temp {max_mem_temp:.0f}C > max threshold {self.cfg.mem_temp_max_c}C"
            )
        return problems

    def check_stress_test_gpuburn(self):
        step = "stress-test-gpuburn"
        name = f"gpu-burn stress test ({self.cfg.gpuburn_duration_s}s)"

        if not self.args.skip_stress:
            gpu_burn = self.cfg.gpu_burn_path
            if not gpu_burn or not shutil.which(gpu_burn):
                self._fail(step, name, f"gpu-burn not found: '{gpu_burn}'")
                return

            print(f"\n  >>> Running gpu-burn -tc for {self.cfg.gpuburn_duration_s}s...")
            print(f"  >>> Monitor in another terminal with:")
            print(f"      nvidia-smi dmon -s puecvmt -d 1")
            print(f"  >>> Or watch for kernel errors with:")
            print(f"      journalctl -k -f | grep -iE 'xid|nvrm|aer'")
            print()

            cmd = [gpu_burn, "-tc", "-d", str(self.cfg.gpuburn_duration_s)]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid,
            )

            monitor = self._monitor_stress(proc, self.cfg.gpuburn_duration_s)
            self.gpuburn_monitor = monitor

            rc = monitor.get("returncode", -1)
            if rc not in (0, None, -15, 124):
                self._fail(
                    step,
                    name,
                    f"gpu-burn exited with code {rc}: {monitor.get('stderr', '')[:200]}",
                    {"rc": rc, "stdout": monitor.get("stdout", "")[:500], "stderr": monitor.get("stderr", "")[:500]},
                )
                return

            out = monitor.get("stdout", "")
            error_lines = [
                line
                for line in out.splitlines()
                if re.search(r"\b(error|fail|fault|uncorrectable)\b", line, re.IGNORECASE)
            ]
            if error_lines:
                self._fail(
                    step,
                    name,
                    f"gpu-burn reported errors: {'; '.join(error_lines[:3])}",
                    {"output": out[:500]},
                )
                return

        info = get_live_metrics()
        gpu_temp = parse_float(info.get("temperature.gpu", "0"))
        mem_temp = parse_float(info.get("temperature.memory", "0"))
        power = read_power_draw(self.cfg.expected_power_max_w)

        max_gpu_temp = gpu_temp
        max_mem_temp = mem_temp
        if hasattr(self, "gpuburn_monitor"):
            max_gpu_temp = max(gpu_temp, self.gpuburn_monitor.get("max_gpu_temp", 0))
            max_mem_temp = max(mem_temp, self.gpuburn_monitor.get("max_mem_temp", 0))

        details_parts = [f"post-burn: gpu_temp={gpu_temp:.0f}C, mem_temp={mem_temp:.0f}C, power={power:.1f}W"]
        if hasattr(self, "gpuburn_monitor") and self.gpuburn_monitor.get("samples", 0) > 0:
            details_parts.append(
                f"during stress: max_gpu_temp={max_gpu_temp:.0f}C, max_mem_temp={max_mem_temp:.0f}C, "
                f"sm_clock={self.gpuburn_monitor['min_sm_clock']:.0f}-{self.gpuburn_monitor['max_sm_clock']:.0f}MHz, "
                f"power={self.gpuburn_monitor['min_power']:.0f}-{self.gpuburn_monitor['max_power']:.0f}W"
            )
        details = "; ".join(details_parts)

        problems = self._eval_stress_thermals(
            {"max_gpu_temp": max_gpu_temp, "max_mem_temp": max_mem_temp}
        )

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", info)
        else:
            self._pass(step, name, details, info)

    def check_stress_test_gpufryer(self):
        step = "stress-test-gpufryer"
        name = f"gpu-fryer stress test ({self.cfg.gpufryer_duration_s}s)"

        if not self.args.skip_stress:
            gpu_fryer = self.cfg.gpu_fryer_path
            if not gpu_fryer or not shutil.which(gpu_fryer):
                self._fail(step, name, f"gpu-fryer not found: '{gpu_fryer}' -- full TGP power/thermal test cannot run")
                return

            print(f"\n  >>> Running gpu-fryer for {self.cfg.gpufryer_duration_s}s...")
            print(f"  >>> Monitor in another terminal with:")
            print(f"      nvidia-smi dmon -s puecvmt -d 1")
            print(f"  >>> Or watch for kernel errors with:")
            print(f"      journalctl -k -f | grep -iE 'xid|nvrm|aer'")
            print()

            rc_help, help_out, _ = run([gpu_fryer, "--help"], timeout=10)
            use_json = rc_help == 0 and "--json" in (help_out or "")

            nvml_path = shutil.which("nvidia-smi") or ""
            nvml_dir = os.path.dirname(nvml_path) if nvml_path else ""
            nvml_lib = ""
            for candidate in [
                "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
                "/usr/lib/libnvidia-ml.so.1",
                "/usr/lib64/libnvidia-ml.so.1",
                os.path.join(nvml_dir, "libnvidia-ml.so.1") if nvml_dir else "",
            ]:
                if candidate and os.path.exists(candidate):
                    nvml_lib = candidate
                    break

            extra_args: list[str] = []
            if use_json:
                extra_args.append("--json")
            if nvml_lib and "--nvml-lib-path" in (help_out or ""):
                extra_args.extend(["--nvml-lib-path", nvml_lib])

            cmd = [gpu_fryer] + extra_args + [str(self.cfg.gpufryer_duration_s)]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid,
            )

            monitor = self._monitor_stress(proc, self.cfg.gpufryer_duration_s)
            self.gpufryer_monitor = monitor

            rc = monitor.get("returncode", -1)
            if rc not in (0, None, -15, 124):
                self._fail(
                    step,
                    name,
                    f"gpu-fryer exited with code {rc}: {monitor.get('stderr', '')[:200]}",
                    {"rc": rc, "stdout": monitor.get("stdout", "")[:500], "stderr": monitor.get("stderr", "")[:500]},
                )
                return

            out = monitor.get("stdout", "")
            gpufryer_problems = self._parse_gpufryer_output(out, use_json=use_json)
            if gpufryer_problems:
                self._fail(
                    step,
                    name,
                    f"gpu-fryer reported problems: {'; '.join(gpufryer_problems)}",
                    {"output": out[:1000]},
                )
                return

        info = get_live_metrics()
        gpu_temp = parse_float(info.get("temperature.gpu", "0"))
        mem_temp = parse_float(info.get("temperature.memory", "0"))
        power = read_power_draw(self.cfg.expected_power_max_w)

        max_gpu_temp = gpu_temp
        max_mem_temp = mem_temp
        if hasattr(self, "gpufryer_monitor"):
            max_gpu_temp = max(gpu_temp, self.gpufryer_monitor.get("max_gpu_temp", 0))
            max_mem_temp = max(mem_temp, self.gpufryer_monitor.get("max_mem_temp", 0))

        details_parts = [f"post-fryer: gpu_temp={gpu_temp:.0f}C, mem_temp={mem_temp:.0f}C, power={power:.1f}W"]
        if hasattr(self, "gpufryer_monitor") and self.gpufryer_monitor.get("samples", 0) > 0:
            details_parts.append(
                f"during stress: max_gpu_temp={max_gpu_temp:.0f}C, max_mem_temp={max_mem_temp:.0f}C, "
                f"sm_clock={self.gpufryer_monitor['min_sm_clock']:.0f}-{self.gpufryer_monitor['max_sm_clock']:.0f}MHz, "
                f"power={self.gpufryer_monitor['min_power']:.0f}-{self.gpufryer_monitor['max_power']:.0f}W, "
                f"vram_used={self.gpufryer_monitor.get('max_vram_used_mib', 0)}MiB"
            )
        details = "; ".join(details_parts)

        problems = self._eval_stress_thermals(
            {"max_gpu_temp": max_gpu_temp, "max_mem_temp": max_mem_temp}
        )

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", info)
        else:
            self._pass(step, name, details, info)

    def _parse_gpufryer_output(self, output: str, use_json: bool = False) -> list[str]:
        """Parse gpu-fryer output for problems (JSON or text mode)."""
        problems: list[str] = []
        if use_json:
            for line in output.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "problems":
                    for p in msg.get("problems", []):
                        problems.append(str(p))
                if msg.get("type") == "results":
                    for gpu in msg.get("gpus", []):
                        burn = gpu.get("burn_result", {})
                        if burn.get("throttling_hw"):
                            problems.append(f"GPU {gpu.get('gpu_idx', '?')}: HW throttling detected")
                        if burn.get("throttling_thermal_sw"):
                            problems.append(f"GPU {gpu.get('gpu_idx', '?')}: SW thermal throttling detected")
                        if burn.get("throttling_thermal_hw"):
                            problems.append(f"GPU {gpu.get('gpu_idx', '?')}: HW thermal throttling detected")
        else:
            if "All GPUs seem healthy" not in output:
                for line in output.splitlines():
                    if re.search(r"\b(error|fail|fault|throttl)\b", line, re.IGNORECASE):
                        problems.append(line.strip())
        return problems

    def check_power_verification(self):
        step = "power-verification"
        name = "Power draw verification under load (gpu-fryer)"
        if self.args.skip_stress or not hasattr(self, "gpufryer_monitor"):
            self._skip(step, name, "No gpu-fryer load monitoring data (stress test skipped or failed)")
            return

        mon = self.gpufryer_monitor
        max_power = mon.get("max_power", 0)
        min_power = mon.get("min_power", 0)

        details = (
            f"during gpu-fryer: power={min_power:.0f}-{max_power:.0f}W, "
            f"TDP={self.cfg.expected_power_default_w}W"
        )

        problems = []
        if self.cfg.expected_power_default_w > 0 and max_power > 0:
            if self.cfg.power_sustain_min_w > 0:
                threshold = self.cfg.power_sustain_min_w
            else:
                threshold = self.cfg.expected_power_default_w * POWER_THRESHOLD_PCT
            if max_power < threshold:
                problems.append(
                    f"max power {max_power:.0f}W < threshold "
                    f"({threshold:.0f}W) -- card not drawing expected power"
                )

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", mon)
        else:
            self._pass(step, name, details, mon)

    def check_vram_fill_verification(self):
        step = "vram-fill-verification"
        name = "VRAM fill verification under load (gpu-fryer)"
        if self.args.skip_stress or not hasattr(self, "gpufryer_monitor"):
            self._skip(step, name, "No gpu-fryer load monitoring data (stress test skipped or failed)")
            return

        mon = self.gpufryer_monitor
        max_vram = mon.get("max_vram_used_mib", 0)

        info = query_gpu(["memory.total"]) or {}
        total_vram = parse_int(info.get("memory.total", "0"))

        pct = (max_vram / total_vram * 100) if total_vram > 0 else 0

        details = (
            f"during gpu-fryer: max_vram_used={max_vram}MiB, "
            f"total={total_vram}MiB ({pct:.0f}% fill)"
        )

        problems = []
        if total_vram > 0 and max_vram > 0:
            if pct < VRAM_FILL_THRESHOLD_PCT * 100:
                problems.append(
                    f"VRAM fill {pct:.0f}% < {VRAM_FILL_THRESHOLD_PCT * 100:.0f}% -- gpu-fryer should fill ~95% of VRAM"
                )

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", mon)
        else:
            self._pass(step, name, details, mon)

    def _check_pcie_under_load(self, monitor_attr: str, label: str):
        step = f"pcie-under-load{'-gpufryer' if label == 'gpu-fryer' else ''}"
        if label == "gpu-fryer":
            name = "PCIe link under full TGP load (gpu-fryer)"
            load_desc = "full TGP load"
            skip_prefix = "No gpu-fryer load monitoring data"
        else:
            name = "PCIe link under load"
            load_desc = "load"
            skip_prefix = "No load monitoring data"

        monitor = getattr(self, monitor_attr, None)
        if self.args.skip_stress or monitor is None:
            self._skip(step, name, f"{skip_prefix} (stress test skipped or failed)")
            return

        info = getattr(self, monitor_attr, {}).get("pcie_link_info", {})
        snapshot_note = ""
        if not info:
            info = get_pcie_link_info()
            snapshot_note = " (post-stress snapshot; captured after stress process ended)"
        if not info:
            self._skip(step, name, "Could not retrieve PCIe link info under load")
            return

        gen_current = parse_int(info.get("pcie_gen_current", "0"))
        width_current = parse_int(info.get("pcie_width_current", "0"))
        replays = parse_int(info.get("pcie_replays", "0"))
        replay_rollovers = parse_int(info.get("pcie_replay_rollovers", "0"))

        details = (
            f"under {load_desc} ({label}): gen={gen_current}x{width_current}, "
            f"replays={replays}, rollovers={replay_rollovers}{snapshot_note}"
        )

        problems = []
        if gen_current < self.cfg.expected_pcie_gen:
            problems.append(f"PCIe gen {gen_current} < expected {self.cfg.expected_pcie_gen} under {load_desc}")
        if width_current < self.cfg.expected_pcie_width:
            problems.append(f"PCIe width x{width_current} < expected x{self.cfg.expected_pcie_width} under {load_desc}")
        if replays > 0:
            problems.append(f"PCIe replay counter non-zero: {replays}")
        if replay_rollovers > 0:
            problems.append(f"PCIe replay rollovers non-zero: {replay_rollovers}")

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", info)
        else:
            self._pass(step, name, details, info)

    def check_pcie_under_load(self):
        self._check_pcie_under_load("gpuburn_monitor", "gpu-burn")

    def check_pcie_under_load_gpufryer(self):
        self._check_pcie_under_load("gpufryer_monitor", "gpu-fryer")

    def _check_clock_verification(self, monitor_attr: str, label: str):
        if label == "gpu-fryer":
            step = "clock-verification-gpufryer"
            name = "Clock and throttle verification under full TGP (gpu-fryer)"
            details_prefix = "during gpu-fryer"
            clock_suffix = "under full TGP load"
            throttle_suffix = "under full TGP"
            skip_prefix = "No gpu-fryer load monitoring data"
        else:
            step = "clock-verification"
            name = "Clock verification under load"
            details_prefix = "during stress"
            clock_suffix = "under load"
            throttle_suffix = "during stress"
            skip_prefix = "No load monitoring data"

        monitor = getattr(self, monitor_attr, None)
        if self.args.skip_stress or monitor is None:
            self._skip(step, name, f"{skip_prefix} (stress test skipped or failed)")
            return

        mon = monitor
        min_sm_clock = mon.get("min_sm_clock", 0)
        max_sm_clock = mon.get("max_sm_clock", 0)
        max_power = mon.get("max_power", 0)
        max_gpu_temp = mon.get("max_gpu_temp", 0)
        throttle_activated = mon.get("throttle_activated", [])

        if label == "gpu-fryer":
            min_mem_clock = mon.get("min_mem_clock", 0)
            max_mem_clock = mon.get("max_mem_clock", 0)
            details = (
                f"{details_prefix}: sm_clock={min_sm_clock:.0f}-{max_sm_clock:.0f}MHz, "
                f"mem_clock={min_mem_clock:.0f}-{max_mem_clock:.0f}MHz, "
                f"max_power={max_power:.1f}W, max_gpu_temp={max_gpu_temp:.0f}C, "
                f"throttle_activated={throttle_activated or 'none'}"
            )
        else:
            details = (
                f"{details_prefix}: sm_clock={min_sm_clock:.0f}-{max_sm_clock:.0f}MHz, "
                f"max_power={max_power:.1f}W, max_gpu_temp={max_gpu_temp:.0f}C, "
                f"throttle_activated={throttle_activated or 'none'}"
            )

        problems = []
        if self.cfg.expected_sm_clock_min_mhz > 0 and min_sm_clock > 0:
            if min_sm_clock < self.cfg.expected_sm_clock_min_mhz:
                problems.append(
                    f"min SM clock {min_sm_clock:.0f}MHz < expected minimum "
                    f"{self.cfg.expected_sm_clock_min_mhz:.0f}MHz {clock_suffix}"
                )
        if label == "gpu-fryer":
            if self.cfg.expected_mem_clock_min_mhz > 0 and min_mem_clock > 0:
                if min_mem_clock < self.cfg.expected_mem_clock_min_mhz:
                    problems.append(
                        f"min memory clock {min_mem_clock:.0f}MHz < expected minimum "
                        f"{self.cfg.expected_mem_clock_min_mhz:.0f}MHz {clock_suffix}"
                    )
        if "clocks_event_reasons.sw_power_cap" in throttle_activated:
            pass  # normal: driver capping power at TDP under full load
        if "clocks_event_reasons.hw_thermal_slowdown" in throttle_activated:
            problems.append(f"HW Thermal Slowdown was ACTIVE {throttle_suffix}")
        if "clocks_event_reasons.hw_power_brake_slowdown" in throttle_activated:
            problems.append(f"HW Power Brake was ACTIVE {throttle_suffix}")
        if "clocks_event_reasons.sw_thermal_slowdown" in throttle_activated:
            problems.append(f"SW Thermal Slowdown was ACTIVE {throttle_suffix}")

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", mon)
        else:
            self._pass(step, name, details, mon)

    def check_clock_verification(self):
        self._check_clock_verification("gpuburn_monitor", "gpu-burn")

    def check_clock_verification_gpufryer(self):
        self._check_clock_verification("gpufryer_monitor", "gpu-fryer")

    # ---- Post-stress checks ----

    # ---- LLM tests ----

    def _run_llm_test(
        self,
        step: str,
        name: str,
        model_file: str,
        model_name: str,
        expected_vram_mib: int,
        context_length: int,
        prompt: str,
        port_offset: int = 0,
    ) -> bool:
        if self.args.skip_llm:
            self._skip(step, name, "LLM tests skipped (--skip-llm)")
            return False

        model_path = Path(model_file)
        if not model_path.is_absolute():
            model_path = Path(self.cfg.models_dir) / model_file
        if not model_path.exists():
            self._skip(step, name, f"Model file not found: {model_path}")
            return False

        llama = self.cfg.llama_server_path
        if not llama or not shutil.which(llama):
            self._skip(step, name, f"llama-server not found: '{llama}'")
            return False

        print(f"\n  >>> Loading {model_name}...")
        print(f"  >>> Model: {model_path}")
        print(f"  >>> Context: {context_length}")
        print(f"  >>> Starting llama-server on {self.cfg.llama_server_host}:{self.cfg.llama_server_port + port_offset}")

        port = self.cfg.llama_server_port + port_offset
        cmd = (
            f"{llama} -m {model_path} -ngl 99 -c {context_length} "
            f"--host {self.cfg.llama_server_host} --port {port}"
        )
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )

        time.sleep(LLAMA_SERVER_STARTUP_WAIT_S)

        loaded = False
        for _ in range((LLAMA_SERVER_READY_TIMEOUT_S - LLAMA_SERVER_STARTUP_WAIT_S + 1) // 2):
            if proc.poll() is not None:
                out, err = proc.communicate()
                self._fail(
                    step,
                    name,
                    f"llama-server exited early: {err[:300]}",
                    {"stdout": out[:500], "stderr": err[:500]},
                )
                return False
            rc_check, _, _ = run(f"curl -s http://localhost:{port}/health", timeout=5)
            if rc_check == 0:
                loaded = True
                break
            time.sleep(2)

        if not loaded:
            os.killpg(os.getpgid(proc.pid), 9)
            self._fail(step, name, f"llama-server did not become ready within {LLAMA_SERVER_READY_TIMEOUT_S}s")
            return False

        print(f"  >>> llama-server ready. Sending prompt...")
        payload = {"messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "seed": 42, "max_tokens": 300}
        prompt_json = json.dumps(payload)
        rc, resp, err = run(
            [
                "curl",
                "-s",
                "-X",
                "POST",
                f"http://localhost:{port}/v1/chat/completions",
                "-H",
                "Content-Type: application/json",
                "-d",
                prompt_json,
            ],
            timeout=180,
        )

        vram_info = query_gpu(["memory.used", "memory.total"])
        vram_used = parse_int(vram_info.get("memory.used", "0")) if vram_info else 0

        os.killpg(os.getpgid(proc.pid), 9)
        proc.wait()

        if rc != 0 or not resp:
            self._fail(step, name, f"curl failed: {err[:200]}", {"response": resp[:500]})
            return False

        if "error" in resp.lower() and "content" not in resp.lower():
            self._fail(step, name, f"API returned error: {resp[:200]}", {"response": resp[:500]})
            return False

        if not resp.strip():
            self._fail(step, name, "Empty response from model")
            return False

        response_text = ""
        parse_ok = False
        try:
            resp_obj = json.loads(resp)
            response_text = resp_obj.get("choices", [{}])[0].get("message", {}).get("content", "")
            parse_ok = True
        except (json.JSONDecodeError, IndexError):
            response_text = resp[:500]

        if not parse_ok:
            self._fail(
                step,
                name,
                f"Model response was not valid JSON (possible VRAM corruption): {resp[:200]}",
                {"response": resp[:500]},
            )
            return False

        print(f"\n  >>> Model output ({model_name}):")
        print(f"  {response_text[:500]}")
        self._manual(
            f"{step}-coherence",
            f"LLM coherence review ({model_name})",
            "Review the model output above. It should be coherent, on-topic, and grammatically correct. "
            "If it is garbage, garbled, or off-topic, FAIL this check manually.",
        )

        details = (
            f"model={model_name}, vram_used={vram_used} MiB "
            f"(expected ~{expected_vram_mib} MiB), response_length={len(resp)}"
        )

        problems = []
        if vram_used < expected_vram_mib * 0.8:
            problems.append(
                f"VRAM usage {vram_used} MiB significantly below expected "
                f"{expected_vram_mib} MiB"
            )

        if problems:
            self._fail(step, name, "; ".join(problems) + f" | {details}", {"response": resp[:500]})
        else:
            self._pass(step, name, details, {"response": resp[:500]})

        time.sleep(3)
        return True

    def check_llm_smoke(self):
        self._run_llm_test(
            step="llm-smoke",
            name=f"LLM smoke test ({self.cfg.smoke_model_name})",
            model_file=self.cfg.smoke_model_file,
            model_name=self.cfg.smoke_model_name,
            expected_vram_mib=self.cfg.smoke_expected_vram_mib,
            context_length=self.cfg.smoke_context_length,
            prompt=self.cfg.smoke_prompt,
            port_offset=0,
        )

    def check_llm_fill(self):
        self._run_llm_test(
            step="llm-fill",
            name=f"LLM VRAM-fill test ({self.cfg.fill_model_name})",
            model_file=self.cfg.fill_model_file,
            model_name=self.cfg.fill_model_name,
            expected_vram_mib=self.cfg.fill_expected_vram_mib,
            context_length=self.cfg.fill_context_length,
            prompt=self.cfg.fill_prompt,
            port_offset=1,
        )

    def check_llm_bench(self):
        step = "llm-bench"
        name = "llama-bench performance benchmark"

        if self.args.skip_llm:
            self._skip(step, name, "LLM tests skipped (--skip-llm)")
            return

        bench = self.cfg.llama_bench_path
        if not bench or not shutil.which(bench):
            self._skip(step, name, f"llama-bench not found: '{bench}'")
            return

        model_file = self.cfg.smoke_model_file
        model_path = Path(model_file)
        if not model_path.is_absolute():
            model_path = Path(self.cfg.models_dir) / model_file
        if not model_path.exists():
            self._skip(step, name, f"Model file not found: {model_path}")
            return

        print(f"\n  >>> Running llama-bench on {self.cfg.smoke_model_name}...")
        # Test prompt processing (pp512) and generation (tg128) with 3 repetitions
        cmd = [
            bench,
            "-m",
            str(model_path),
            "-ngl",
            "99",
            "-p",
            "512",
            "-n",
            "128",
            "-r",
            "3",
            "-o",
            "csv",
        ]
        rc, out, err = run(cmd, timeout=300)

        if rc != 0:
            self._fail(step, name, f"llama-bench exited with code {rc}: {err[:200]}")
            return

        if not out.strip():
            self._fail(step, name, "llama-bench produced no output")
            return

        # Parse CSV output: header line + data lines
        # llama-bench CSV columns (recent versions):
        #   ...,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts
        lines = [l for l in out.splitlines() if l.strip()]
        if len(lines) < 2:
            self._fail(step, name, f"llama-bench output too short: {out[:200]}")
            return

        header = lines[0].split(",")
        # Find column indices by header name
        col_map = {name.strip(): i for i, name in enumerate(header)}

        n_prompt_col = col_map.get("n_prompt")
        n_gen_col = col_map.get("n_gen")
        avg_ts_col = col_map.get("avg_ts")

        if avg_ts_col is None:
            self._skip(
                step, name,
                f"Could not find avg_ts column in llama-bench output (check llama-bench version)",
            )
            return

        pp_vals: list[float] = []
        tg_vals: list[float] = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= avg_ts_col:
                continue
            try:
                avg_ts = float(parts[avg_ts_col].strip('"'))
            except ValueError:
                continue

            n_prompt = ""
            n_gen = ""
            if n_prompt_col is not None and n_prompt_col < len(parts):
                n_prompt = parts[n_prompt_col].strip('"')
            if n_gen_col is not None and n_gen_col < len(parts):
                n_gen = parts[n_gen_col].strip('"')

            if n_prompt and n_prompt != "0":
                pp_vals.append(avg_ts)
            elif n_gen and n_gen != "0":
                tg_vals.append(avg_ts)

        if not pp_vals or not tg_vals:
            self._skip(
                step, name,
                f"Could not parse benchmark results (check llama-bench output format)",
            )
            return

        pp_avg = sum(pp_vals) / len(pp_vals)
        tg_avg = sum(tg_vals) / len(tg_vals)

        details = (
            f"pp512={pp_avg:.1f} tokens/s, tg128={tg_avg:.1f} tokens/s "
            f"(model: {self.cfg.smoke_model_name})"
        )

        # Store for baseline comparison
        self.proto.log_info(
            f"PERF: pp512={pp_avg:.1f} tg128={tg_avg:.1f} model={self.cfg.smoke_model_name}"
        )
        self._pass(step, name, details, {"pp_avg": pp_avg, "tg_avg": tg_avg, "raw": out[:1000]})

        if self.cfg.smoke_perf_baseline_label and self.cfg.smoke_perf_tolerance_pct > 0:
            self._manual(
                f"{step}-perf",
                f"Performance baseline comparison ({self.cfg.smoke_perf_baseline_label})",
                f"Compare pp512={pp_avg:.1f} and tg128={tg_avg:.1f} against the {self.cfg.smoke_perf_baseline_label} baseline. "
                f"If performance is more than {self.cfg.smoke_perf_tolerance_pct:.0f}% below baseline, FAIL this check.",
            )
        else:
            self._pass(
                f"{step}-perf",
                f"Performance baseline comparison (none configured)",
                "No baseline label or tolerance configured; cannot compare performance automatically.",
                {"pp_avg": pp_avg, "tg_avg": tg_avg},
            )

    # ---- Final checks ----

    def check_cooldown(self):
        step = "cooldown"
        name = f"Post-test cooldown ({self.cfg.cooldown_time_s}s)"

        if self.args.skip_stress:
            self._skip(step, name, "Skipped (no stress test)")
            return

        print(f"\n  >>> Waiting {self.cfg.cooldown_time_s}s for cooldown...")
        time.sleep(self.cfg.cooldown_time_s)

        info = get_live_metrics()
        gpu_temp = parse_float(info.get("temperature.gpu", "0"))
        power = read_power_draw(self.cfg.expected_power_max_w)

        details = f"gpu_temp={gpu_temp:.0f}C, power={power:.1f}W after {self.cfg.cooldown_time_s}s"

        if gpu_temp > self.cfg.cooldown_temp_max_c:
            self._fail(
                step,
                name,
                f"GPU temp {gpu_temp:.0f}C > {self.cfg.cooldown_temp_max_c}C after cooldown "
                f"(cooling system may be degraded)",
                info,
            )
        else:
            self._pass(step, name, details, info)

    def check_bug_report(self):
        step = "bug-report"
        name = "nvidia-bug-report capture"

        if self.args.skip_bug_report:
            self._skip(step, name, "Skipped (--skip-bug-report)")
            return

        bug_report = shutil.which("nvidia-bug-report.sh")
        if not bug_report:
            self._skip(step, name, "nvidia-bug-report.sh not found in PATH")
            return

        print("\n  >>> Running nvidia-bug-report.sh (may take a minute)...")
        # Try without sudo first (works if driver is accessible), then with sudo
        rc, out, err = run("nvidia-bug-report.sh", timeout=300)
        if rc != 0:
            print("  >>> Retrying with sudo...")
            rc, out, err = run("sudo nvidia-bug-report.sh", timeout=300)

        if rc != 0:
            self._skip(
                step, name,
                f"nvidia-bug-report failed (needs sudo or root). Not a card health issue.",
            )
        else:
            log_file = "nvidia-bug-report.log.gz"
            if Path(log_file).exists():
                self._pass(step, name, f"Saved to {log_file}")
            else:
                self._skip(
                    step, name,
                    f"Completed but output file not found. Not a card health issue.",
                )

    # ---- Orchestrator ----

    def run_all(self):
        checks = [
            self.check_identity,
            self.check_power_limits,
            self.check_pcie_link,
            lambda: self._check_ecc_state("ecc-baseline", "ECC status (baseline)", "baseline"),
            lambda: self._check_row_remapper("row-remapper-baseline", "Row remapper status (baseline)", "baseline"),
            lambda: self._check_retired_pages("retired-pages-baseline", "Retired pages (baseline)", "baseline"),
            lambda: self._check_aer("aer-baseline", "PCIe AER counters (baseline)", "baseline", store_baseline=True),
            self.check_idle_thermals,
            self.check_tlimit_thresholds,
            lambda: self.check_kernel_log("kernel-log-baseline", "Kernel log scan (baseline)", "baseline"),
            self.check_stress_test_gpuburn,
            self.check_pcie_under_load,
            self.check_clock_verification,
            self.check_stress_test_gpufryer,
            self.check_power_verification,
            self.check_vram_fill_verification,
            self.check_pcie_under_load_gpufryer,
            self.check_clock_verification_gpufryer,
            lambda: self._check_ecc_state("ecc-post", "ECC status (post-stress)", "post-stress"),
            lambda: self._check_row_remapper("row-remapper-post", "Row remapper status (post-stress)", "post-stress"),
            lambda: self._check_retired_pages("retired-pages-post", "Retired pages (post-stress)", "post-stress"),
            lambda: self._check_aer_with_delta("aer-post", "PCIe AER counters (post-stress)", "post-stress", "post", "Correctable PCIe errors are self-healed, but >10 new during stress suggests signal integrity issues. Try reseating the card in the slot and re-running. If the count still rises, walk away (slot/riser/retimer problem). If it stays at 0 after reseat, the original count was likely from insertion, safe to proceed."),
            lambda: self.check_kernel_log("kernel-log-post-stress", "Kernel log scan (post-stress)", "post-stress"),
            self.check_llm_smoke,
            self.check_llm_fill,
            self.check_llm_bench,
            lambda: self._check_ecc_state("ecc-final", "ECC status (final)", "final"),
            lambda: self._check_row_remapper("row-remapper-final", "Row remapper status (final)", "final"),
            lambda: self._check_retired_pages("retired-pages-final", "Retired pages (final)", "final"),
            lambda: self._check_aer_with_delta("aer-final", "PCIe AER counters (final)", "final", "final", "Same as post-stress warning but over the full session. If the post-stress check already passed after reseat, this is cumulative and safe to ignore. If not previously addressed, reseat and re-run."),
            lambda: self.check_kernel_log("kernel-log-final", "Kernel log scan (final)", "final"),
            self.check_cooldown,
            self.check_bug_report,
        ]

        for check_fn in checks:
            check_fn()

        self._print_walkaway_reminder()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _coerce_int(d: dict, key: str, default: int = 0, section: str = "") -> int:
    v = d.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        loc = f"{section}.{key}" if section else key
        raise SystemExit(f"Config error: {loc} must be an integer, got {v!r}")


def _coerce_float(d: dict, key: str, default: float = 0, section: str = "") -> float:
    v = d.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        loc = f"{section}.{key}" if section else key
        raise SystemExit(f"Config error: {loc} must be a number, got {v!r}")


def _coerce_bool(d: dict, key: str, default: bool = False, section: str = "") -> bool:
    v = d.get(key, default)
    if isinstance(v, bool):
        return v
    loc = f"{section}.{key}" if section else key
    raise SystemExit(f"Config error: {loc} must be a boolean, got {v!r}")


def load_config(config_path: Path) -> GPUConfig:
    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    card = data.get("card", {})
    thresholds = data.get("thresholds", {})
    stress = data.get("stress", {})
    llm_smoke = data.get("llm", {}).get("smoke", {})
    llm_fill = data.get("llm", {}).get("fill", {})
    paths = data.get("paths", {})

    return GPUConfig(
        name=card.get("name", "unknown"),
        expected_gpu_name=card.get("expected_gpu_name", ""),
        expected_vram_mib=_coerce_int(card, "expected_vram_mib", 0, "card"),
        expected_device_id=card.get("expected_device_id", ""),
        expected_subsystem_vendor=card.get("expected_subsystem_vendor", ""),
        vbios_prefix=card.get("vbios_prefix", ""),
        expected_power_default_w=_coerce_float(card, "expected_power_default_w", 0, "card"),
        expected_power_max_w=_coerce_float(card, "expected_power_max_w", 0, "card"),
        expected_pcie_gen=_coerce_int(card, "expected_pcie_gen", 0, "card"),
        expected_pcie_width=_coerce_int(card, "expected_pcie_width", 0, "card"),
        ecc_supported=_coerce_bool(card, "ecc_supported", False, "card"),
        ecc_expected_enabled=_coerce_bool(card, "ecc_expected_enabled", False, "card"),
        gpu_temp_max_c=_coerce_float(thresholds, "gpu_temp_max_c", 85, "thresholds"),
        gpu_temp_walkaway_c=_coerce_float(thresholds, "gpu_temp_walkaway_c", 90, "thresholds"),
        mem_temp_max_c=_coerce_float(thresholds, "mem_temp_max_c", 95, "thresholds"),
        mem_temp_walkaway_c=_coerce_float(thresholds, "mem_temp_walkaway_c", 100, "thresholds"),
        power_sustain_min_w=_coerce_float(thresholds, "power_sustain_min_w", 0, "thresholds"),
        idle_temp_max_c=_coerce_float(thresholds, "idle_temp_max_c", 55, "thresholds"),
        idle_power_max_w=_coerce_float(thresholds, "idle_power_max_w", 30, "thresholds"),
        cooldown_time_s=_coerce_int(thresholds, "cooldown_time_s", 120, "thresholds"),
        cooldown_temp_max_c=_coerce_float(thresholds, "cooldown_temp_max_c", 60, "thresholds"),
        gpuburn_duration_s=_coerce_int(stress, "gpuburn_duration_s", 300, "stress"),
        gpufryer_duration_s=_coerce_int(stress, "gpufryer_duration_s", 600, "stress"),
        expected_sm_clock_min_mhz=_coerce_float(card, "expected_sm_clock_min_mhz", 0, "card"),
        expected_mem_clock_min_mhz=_coerce_float(card, "expected_mem_clock_min_mhz", 0, "card"),
        smoke_model_file=llm_smoke.get("model_file", ""),
        smoke_model_name=llm_smoke.get("name", "smoke test"),
        smoke_expected_vram_mib=_coerce_int(llm_smoke, "expected_vram_usage_mib", 0, "llm.smoke"),
        smoke_context_length=_coerce_int(llm_smoke, "context_length", 4096, "llm.smoke"),
        smoke_perf_baseline_label=llm_smoke.get("perf_baseline_label", ""),
        smoke_perf_tolerance_pct=_coerce_float(llm_smoke, "perf_tolerance_pct", 0, "llm.smoke"),
        fill_model_file=llm_fill.get("model_file", ""),
        fill_model_name=llm_fill.get("name", "fill test"),
        fill_expected_vram_mib=_coerce_int(llm_fill, "expected_vram_usage_mib", 0, "llm.fill"),
        fill_context_length=_coerce_int(llm_fill, "context_length", 8192, "llm.fill"),
        models_dir=paths.get("models_dir", os.environ.get("GPU_CHECK_MODELS_DIR", ".")),
        gpu_burn_path=paths.get("gpu_burn_path", os.environ.get("GPU_CHECK_GPU_BURN", "gpu-burn")),
        gpu_fryer_path=paths.get("gpu_fryer_path", os.environ.get("GPU_CHECK_GPU_FRYER", "gpu-fryer")),
        llama_server_path=paths.get(
            "llama_server_path", os.environ.get("GPU_CHECK_LLAMA_SERVER", "llama-server")
        ),
        llama_bench_path=paths.get(
            "llama_bench_path", os.environ.get("GPU_CHECK_LLAMA_BENCH", "llama-bench")
        ),
        llama_server_host=paths.get("llama_server_host", "127.0.0.1"),
        llama_server_port=_coerce_int(paths, "llama_server_port", 8080, "paths"),
        smoke_prompt=llm_smoke.get(
            "prompt",
            "Explain how a transformer neural network processes an input sequence, "
            "from tokenisation to the final output logits. Cover embedding, attention, "
            "and the feed-forward layers. Aim for about 200 words.",
        ),
        fill_prompt=llm_fill.get(
            "prompt",
            "Explain how a transformer neural network processes an input sequence, "
            "from tokenisation to the final output logits. Cover embedding, attention, "
            "and the feed-forward layers. Aim for about 200 words.",
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="GPU pre-purchase health check tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  gpu-check.py --config configs/rtx-4070-laptop.toml --continue-on-fail\n"
            "  gpu-check.py --config configs/rtx-pro-6000-maxq.toml\n"
            "\n"
            "Environment variables for path overrides:\n"
            "  GPU_CHECK_MODELS_DIR   Directory containing GGUF model files\n"
            "  GPU_CHECK_GPU_BURN     Path to gpu_burn binary\n"
            "  GPU_CHECK_GPU_FRYER    Path to gpu-fryer binary\n"
            "  GPU_CHECK_LLAMA_SERVER Path to llama-server binary\n"
            "  GPU_CHECK_LLAMA_BENCH  Path to llama-bench binary\n"
        ),
    )
    parser.add_argument(
        "--config",
        required=False,
        type=Path,
        default=None,
        help="Path to TOML config file for the target GPU",
    )
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue with the next check even if one fails (useful for testing)",
    )
    parser.add_argument(
        "--skip-stress",
        action="store_true",
        help="Skip the gpu-burn and gpu-fryer stress tests",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM inference tests",
    )
    parser.add_argument(
        "--skip-bug-report",
        action="store_true",
        help="Skip nvidia-bug-report.sh capture",
    )
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="List all check names and exit",
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("protocols"),
        help="Directory for protocol output files (default: protocols)",
    )

    args = parser.parse_args()

    if args.list_checks:
        print("Available checks:")
        for cid, cname in CHECK_NAMES:
            print(f"  {cid:30s}  {cname}")
        sys.exit(0)

    if not args.config:
        parser.error("--config is required unless --list-checks is used")

    config_path = args.config.resolve()
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)

    protocol_dir = args.protocol_dir.resolve()
    protocol_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_stem = config_path.stem
    protocol_path = protocol_dir / f"protocol_{config_stem}_{ts}.md"

    protocol = Protocol(protocol_path)
    protocol.log_info(f"Card: {config.name}")
    protocol.log_info(f"Config: {config_path}")
    protocol.log_info(f"Protocol: {protocol_path}")
    protocol.log_info(f"Continue on fail: {args.continue_on_fail}")
    protocol.log_info("")

    if not shutil.which("nvidia-smi"):
        protocol.log(
            CheckResult(
                step="prerequisite",
                name="nvidia-smi availability",
                status="FAIL",
                details="nvidia-smi not found in PATH. Is the NVIDIA driver installed?",
            )
        )
        sys.exit(1)
    else:
        protocol.log(
            CheckResult(
                step="prerequisite",
                name="nvidia-smi availability",
                status="PASS",
                details=shutil.which("nvidia-smi"),
            )
        )

    checker = GPUChecker(config, protocol, args)

    try:
        checker.run_all()
    except KeyboardInterrupt:
        protocol.log_info("\n*** Interrupted by user ***")
    finally:
        protocol.summary()
        protocol.close()
        print(f"\nProtocol saved to: {protocol_path}")


if __name__ == "__main__":
    main()
