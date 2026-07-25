# GPU Check Tool

Pre-purchase health check tool for used NVIDIA GPUs. Runs a structured
sequence of diagnostic checks and writes a pass/fail protocol for each step.
Designed for verifying used GPUs before buying, but also useful for general
health checking.

## Usage

```bash
# List all checks
python3 gpu-check.py --list-checks

# Full run (stops on first failure)
python3 gpu-check.py --config configs/rtx-pro-6000-maxq.toml

# Continue on fail (for testing the tool itself)
python3 gpu-check.py --config configs/rtx-4070-laptop.toml --continue-on-fail

# Quick check (no stress, no LLM, no bug-report, no DCGM)
python3 gpu-check.py --config configs/rtx-4070-laptop.toml \
  --skip-stress --skip-llm --skip-bug-report --skip-dcgm
```

### CLI flags

| Flag | Effect |
|---|---|
| `--config FILE` | Path to TOML config for the target GPU (required) |
| `--continue-on-fail` | Keep running after a failure (default: stop) |
| `--skip-stress` | Skip gpu-burn stress test |
| `--skip-llm` | Skip LLM inference tests |
| `--skip-bug-report` | Skip nvidia-bug-report capture |
| `--skip-dcgm` | Skip DCGM diagnostic |
| `--list-checks` | List all check names and exit |
| `--protocol-dir DIR` | Directory for protocol output (default: protocols) |

### Environment variables (override config `[paths]` section)

| Variable | Purpose |
|---|---|
| `GPU_CHECK_MODELS_DIR` | Directory containing GGUF model files |
| `GPU_CHECK_GPU_BURN` | Path to gpu_burn binary |
| `GPU_CHECK_LLAMA_SERVER` | Path to llama-server binary |
| `GPU_CHECK_LLAMA_BENCH` | Path to llama-bench binary |
| `GPU_CHECK_DCGMI` | Path to dcgmi binary |

## Config files

Each TOML config has:
- `[card]`: Expected GPU identity (name, VRAM, device ID, VBIOS prefix, power
  limits, PCIe gen/width, ECC support, min clock speeds under load)
- `[thresholds]`: Temperature and power limits for pass/fail
- `[stress]`: gpu-burn duration
- `[llm.smoke]`: Small model for basic CUDA test
- `[llm.fill]`: Large model to fill VRAM
- `[paths]`: Machine-specific binary paths and models directory

`model_file` can be an absolute path or a filename relative to `models_dir`.

## Checks performed (31 total)

### Identity and baseline (before stress)

1. nvidia-smi availability
2. GPU identity (name, VRAM, device ID, VBIOS, serial, board part number)
3. Power limits (detect shunt mods / VBIOS tampering)
4. PCIe link (gen, width, replay errors)
5. Thermal T.Limit threshold sanity (detect VBIOS corruption)
6. ECC status and error counts
7. Row remapper status (prior contained faults)
8. Retired pages (prior contained faults)
9. PCIe AER counters (sysfs correctable, nonfatal, fatal)
10. Idle thermals and power
11. Kernel log scan for Xid errors
12. DCGM diagnostic (if dcgmi available)

### Stress test

13. gpu-burn stress test (default 15 min)
14. PCIe link under load (auto-resolves idle PCIe gen warning)
15. Clock verification under load (screens for SW Power Cap telemetry bug)

### Post-stress

16. ECC error counts (delta from baseline)
17. Row remapper status
18. Retired pages status
19. PCIe AER counters (delta from baseline, with error type breakdown)
20. Kernel log scan
21. Clock throttle reasons (HW/SW thermal, HW power brake, SW power cap)

### LLM tests

22. LLM smoke test (load small model, verify coherent output, check VRAM)
23. LLM VRAM-fill test (load large model, verify VRAM fills correctly)
24. llama-bench benchmark (pp512 + tg128 tokens/sec)

### Final

25-29. ECC, row remapper, retired pages, AER (delta), kernel log
30. Cooldown verification
31. nvidia-bug-report capture

## Understanding the protocol output

Each check is logged as `[PASS]`, `[FAIL]`, `[WARN]`, `[SKIP]`, or `[MANUAL]`.

### WARN results

Warnings include an `ACTION:` line with explicit instructions:

- **PCIe gen low at idle**: Normal power saving. The `pcie-under-load` check
  re-verifies under stress. If that passes, ignore this warning.
- **AER correctable errors during stress**: Correctable PCIe errors are
  self-healed. If the count is low and stable after reseating, safe to
  proceed. If the count keeps rising after reseat, walk away.
- **T.Limit specification at zero**: Possible VBIOS thermal config corruption
  (known Blackwell bug). Compare with a known-good card of the same model.
- **DCGM warnings**: Review individually. Tooling issues are irrelevant to
  the buy decision; PCIe or memory warnings need investigation.
- **nvidia-bug-report / llama-bench parse failures**: Tooling issues, not
  card health. Re-run manually if needed.

### FAIL results

Any FAIL means **do not buy**. The script stops at the first failure unless
`--continue-on-fail` is set. There is no return window on a private sale.

### Walk-away conditions

- Any ECC error (single bit, double bit, DRAM correctable/uncorrectable)
- Any remapped rows or retired pages (prior contained faults)
- Any Xid error in kernel log
- Any AER non-fatal or fatal error
- gpu-burn failure, crash, or error output
- GPU temp above walkaway threshold
- Memory temp above walkaway threshold
- HW Thermal Slowdown, HW Power Brake, or SW Thermal Slowdown active
- SW Power Cap active under load (known Blackwell telemetry bug)
- PCIe link degradation under load (gen, width, or replay errors)
- PCIe width below expected (slot/riser problem)
- Power limit mismatch (VBIOS tampering or shunt mod)
- LLM output incoherent or VRAM does not fill to expected level
- Performance more than 15 percent below baseline

**If in doubt, walk away.**

## Dependencies

### Required

- Python 3.11+ (uses `tomllib`)
- NVIDIA driver + `nvidia-smi`
- Linux (uses sysfs for AER, journalctl/dmesg for kernel logs)

### For stress test

- [gpu-burn](https://github.com/wilicc/gpu-burn) (`gpu_burn` binary in PATH or
  set path in config / env var)

### For LLM tests

- [llama.cpp](https://github.com/ggerganov/llama.cpp) with CUDA support
  (`llama-server` and `llama-bench` binaries)
- GGUF model files (smoke test model + VRAM-fill model)

### Optional

- [DCGM](https://developer.nvidia.com/dcgm) (`dcgmi` binary) for NVIDIA's
  own diagnostic
- `nvidia-bug-report.sh` (ships with driver) for bug report capture
- `sudo` access (for nvidia-bug-report if driver is not accessible without)

## Config files included

- `configs/rtx-pro-6000-maxq.toml`: RTX Pro 6000 Blackwell Max-Q (96 GB,
  Dell OEM variant)
- `configs/rtx-5090.toml`: RTX 5090 (32 GB, for baseline comparison)
- `configs/rtx-4070-laptop.toml`: RTX 4070 Laptop (8 GB, tested config)

## Using the tool on a different card

1. Create a new TOML config in `configs/` (copy an existing one)
2. Set the expected card values (name, VRAM, device ID, power limits, PCIe
   gen/width, ECC support, min clock speeds under load)
3. Set the threshold values (max temps, idle power, cooldown)
4. Set the stress test duration
5. Set the LLM model files and expected VRAM usage
6. Set the `[paths]` section for your machine
7. Run `--list-checks` to verify, then run the full check

## License

MIT
