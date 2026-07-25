# RTX Pro 6000 Blackwell Max-Q: Pre-Purchase Test Plan

## Overview

This document describes the full procedure for verifying a used RTX Pro 6000
Blackwell Max-Q Workstation Edition GPU before purchase. It pairs with the
automated `gpu-check.py` script, which handles the software checks. The manual
steps below cover physical inspection and human judgement that the script
cannot perform.

**Philosophy**: This is a major investment with no return window. Every check
is pass/fail. If anything is even slightly fishy, walk away. There will be
other cards.

**Time budget**: 45 minutes total (35 min automated + 5 min physical + 5 min buffer).

---

## Before the Seller Arrives: Pre-Staging

Complete all of the following on the test workstation in advance.

### 1. Get the GPU check tool

```bash
git clone <repo-url> gpu-check
cd gpu-check
```

Verify the script runs:

```bash
python3 gpu-check.py --list-checks
```

### 2. Install gpu-burn

```bash
# Option A: use a pre-built binary
# Download from: https://github.com/wilicc/gpu-burn/releases
# Place the binary in your PATH or set GPU_CHECK_GPU_BURN env var

# Option B: build from source (requires CUDA toolkit)
git clone https://github.com/wilicc/gpu-burn.git
cd gpu-burn
make
export GPU_CHECK_GPU_BURN="$PWD/gpu_burn"
cd ..
```

Verify:

```bash
gpu_burn -h
```

### 3. Install gpu-fryer

```bash
# Option A: cargo (native binary)
cargo install gpu-fryer
export GPU_CHECK_GPU_FRYER="$HOME/.cargo/bin/gpu-fryer"

# Option B: Docker / Podman (no build needed)
# No install required, but set path in config to the container run command
# Or just install the binary for simplicity
```

Verify:

```bash
gpu-fryer --version
```

### 4. Install llama.cpp

You need `llama-server` and `llama-bench` with CUDA support.

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make GGML_CUDA=1 -j
export GPU_CHECK_LLAMA_SERVER="$PWD/llama-server"
export GPU_CHECK_LLAMA_BENCH="$PWD/llama-bench"
cd ..
```

Verify:

```bash
llama-server --help | head -5
llama-bench --help | head -5
```

### 5. Download model files

Download these GGUF files. The script needs the first shard of split files;
llama-server auto-loads the rest.

| Model | File | Size | Role |
|---|---|---|---|
| Qwen3.6-27B Q4_K_M | `qwen3.6-27b-q4_k_m.gguf` | ~18 GB | Smoke test (daily driver) |
| Qwen3.5-122B-A10B Q4_K_M | `Qwen_Qwen3.5-122B-A10B-Q4_K_M-00001-of-00002.gguf` | ~67 GB (2 shards) | VRAM-fill test |

Place them in a directory and note the path.

### 6. Configure the config file

Open `configs/rtx-pro-6000-maxq.toml`. The **only section you need to change**
is `[paths]` at the bottom. Set `models_dir` to your GGUF directory, and set
`gpu_burn_path`, `gpu_fryer_path`, `llama_server_path`, and `llama_bench_path`
to the full paths of each binary if they are not in PATH.

Alternatively, use environment variables (these override the config):

```bash
export GPU_CHECK_MODELS_DIR="/path/to/your/gguf/files"
export GPU_CHECK_GPU_BURN="/full/path/to/gpu_burn"
export GPU_CHECK_GPU_FRYER="/full/path/to/gpu-fryer"
export GPU_CHECK_LLAMA_SERVER="/full/path/to/llama-server"
export GPU_CHECK_LLAMA_BENCH="/full/path/to/llama-bench"
```

### 7. Pre-compute 5090 baseline (if workstation has a 5090)

Before the seller arrives, run the full check on the 5090 to record a
performance baseline:

```bash
# Edit configs/rtx-5090.toml [paths] section first!
python3 gpu-check.py --config configs/rtx-5090.toml --continue-on-fail
```

Note: the 5090 has 32 GB VRAM and cannot load the 67 GB fill model. The
`llm-fill` check will SKIP if the model file is absent, or FAIL if present.
Use `--skip-llm` if you only need the stress test baseline, or let it SKIP.

Note the `pp512` and `tg128` values from the llama-bench step. The Max-Q
should be within 10-15% of these (both cards have 1792 GB/s bandwidth).

### 8. Verify the tool works

Run the quick check (no stress, no LLM) to confirm all tooling is working:

```bash
python3 gpu-check.py --config configs/rtx-5090.toml --skip-stress --skip-llm --skip-bug-report
```

### 9. Have ready

- Phone or camera for photographing the card
- Good light source (phone flashlight or desk lamp)
- The config file: `configs/rtx-pro-6000-maxq.toml` with `[paths]` set
- If you need sudo for nvidia-bug-report, ensure passwordless sudo or run
  the script as root

---

## Phase 1: Physical Inspection (3 min)

Perform these checks with the card disconnected from all power, held in good
light. The seller should understand this is standard procedure.

### 1.1 16-pin power connector

Look into the 16-pin (12V-2x6) power socket on the card edge.

**What to look for:**
- Scorch marks or brown discolouration on the plastic
- Darkened or blackened pins
- Pins pushed back into the housing
- Loose or wobbly connector housing
- Burnt plastic smell near the connector

**PASS**: Connector is clean, all pins are uniform in colour and fully seated.
**WALK AWAY**: Any scorch, darkening, pushed pins, or burnt smell. This
connector family has documented ongoing melting failures. A damaged connector
is a hard no.

### 1.2 PCIe edge fingers

Examine the gold contact fingers on the bottom edge of the card.

**What to look for:**
- Deep scratches or scoring
- Pitting or corrosion
- Burnt marks
- Dirt between contacts
- Uneven wear (some contacts much more worn than others)

**PASS**: Clean, even gold contacts with normal wear.
**WALK AWAY**: Deep scratches, corrosion, pitting, or burnt marks.

### 1.3 Screws and warranty stickers

Check all screws on the card, especially those holding the cooler/shroud.

**What to look for:**
- Missing screws
- Stripped screw heads
- Mismatched screws (different types/sizes mixed in)
- Torn or damaged warranty stickers
- Signs the cooler has been removed

**PASS**: All screws present, matching, undamaged. Warranty stickers intact.
**WALK AWAY**: Missing/stripped/mismatched screws, torn warranty stickers.
A card that has been opened carries unknown risk.

### 1.4 Fans

With the card disconnected from power, gently spin each fan with your finger.

**What to look for:**
- Grinding, clicking, or scraping sounds
- Stiff resistance or uneven rotation
- Fan wobble or loose bearings
- Cracked or damaged blades
- One fan significantly stiffer than others

**PASS**: Fans spin freely and smoothly, stop gradually.
**WALK AWAY**: Any grinding, clicking, stiffness, wobble, or blade damage.

### 1.5 PCB condition

Examine the printed circuit board, both front and back.

**What to look for:**
- Darkened or discoloured areas, especially near VRM power stages
- PCB warping (slight sag is normal, obvious warp is not)
- Flux residue around components (sign of repair work)
- Corrosion or liquid residue
- Green/blue residue (corrosion from moisture)
- Burnt marks anywhere
- Lifted or damaged solder joints

**PASS**: Clean PCB, no discolouration, no corrosion, no repair marks.
**WALK AWAY**: Any discolouration, corrosion, flux residue, warping, or
burnt marks.

### 1.6 DisplayPort outputs

Examine the four DisplayPort 2.1b connectors.

**What to look for:**
- Bent metal shields
- Loose connectors
- Missing plastic internal tabs
- Damaged pins inside
- Signs of forced cable insertion

**PASS**: All ports clean, intact, firmly seated.
**WALK AWAY**: Bent, loose, or damaged ports.

### 1.7 Heatsink fins

Examine the heatsink fins through the cooler openings.

**What to look for:**
- Bent or crushed fins (suggests rough handling)
- Heavy dust packing (suggests long service without cleaning)
- Oily residue

**PASS**: Fins reasonably clean and undamaged.
**WALK AWAY**: Heavy fin damage or packed dust indicating long heavy use.

### 1.8 Serial label

Locate the serial number label on the card (near the DisplayPorts or on the
back).

**What to look for:**
- Label is present and legible
- Serial number is not defaced or removed
- Note the serial number for warranty purposes

**PASS**: Serial label present, legible, undamaged.
**WALK AWAY**: Missing, defaced, or removed serial label.

### 1.9 Smell test

Hold the card near your nose (especially near the power connector and VRM
area).

**PASS**: Normal electronics smell or no smell.
**WALK AWAY**: Burnt plastic or acrid smell. This indicates prior overheating
damage.

### 1.10 Photograph the card

Take clear photos of:
- Front of card (shroud and fans)
- Back of card (backplate or PCB)
- 16-pin power connector (close-up)
- PCIe edge fingers
- Serial number label
- DisplayPort area

Keep these as evidence of the card's condition at time of purchase.

---

## Phase 2: Install and Run Automated Checks (35 min)

### 2.1 Install the card

1. Power off the workstation
2. Install the card in a direct PCIe 5.0 x16 slot
3. Connect the 16-pin power cable directly (no adapters if possible)
4. Ensure the power cable is fully seated (push until it clicks)
5. Power on

### 2.2 Run the automated check

```bash
cd gpu-check
python3 gpu-check.py --config configs/rtx-pro-6000-maxq.toml
```

**Without `--continue-on-fail`**, the script will stop at the first failure.
This is the correct mode for the actual purchase test. Use
`--continue-on-fail` only when testing the tool itself.

The script runs 36 checks in sequence:

1. nvidia-smi availability
2. GPU identity (name, VRAM, device ID, VBIOS, serial)
3. Power limits (detect shunt mods / VBIOS tampering)
4. PCIe link (gen, width, replay errors)
5. Thermal T.Limit threshold sanity check
6-11. Baseline: ECC, row remapper, retired pages, AER counters, idle thermals, kernel log
12. gpu-burn stress test (5 min, ALU error detection, clock verification)
13. PCIe link under load (gpu-burn)
14. Clock verification under load
15. gpu-fryer stress test (10 min, full TGP power/thermal, VRAM fill)
16. Power draw verification under full TGP load
17. VRAM fill verification
18. PCIe link under full TGP load (gpu-fryer)
19. Clock and throttle verification under full TGP (gpu-fryer)
20-25. Post-stress: ECC, row remapper, retired pages, AER counters, AER error delta, kernel log
26. LLM smoke test (load model, verify coherent output)
27. LLM VRAM-fill test (load large model, verify VRAM usage)
28. llama-bench benchmark (pp512 and tg128 tokens/sec)
29-34. Final: ECC, row remapper, retired pages, AER counters, AER error delta, kernel log
35. Cooldown verification
36. nvidia-bug-report capture

### 2.3 During the stress tests

The script runs two stress tests for a total of 15 minutes:

1. **gpu-burn** (5 min): ALU error detection, clock verification, PCIe under
   moderate load
2. **gpu-fryer** (10 min): Full TGP power/thermal stress, VRAM fill to ~95%,
   PCIe under maximum power draw

During both tests:

1. **Listen** for fan grinding, clicking, or coil whine
2. **Watch** for any visual artefacts on screen
3. **Smell** for any burning odour
4. The script monitors temperatures, power, clocks, ECC, and kernel logs
   automatically

If you notice any of the above, stop the test immediately and walk away.

### 2.4 During the LLM tests

The script will load two models, send a test prompt, and run llama-bench.
Verify manually that:
1. The model output is coherent (not garbage characters)
2. The response is correct for the test prompt
3. The llama-bench pp512 and tg128 values are within 10-15% of the 5090
   baseline (if available)

### 2.5 Review the protocol

After the script completes, review the generated protocol file (in
`protocols/`). Every check should show `[PASS]`. If any check shows `[FAIL]`,
walk away. If any check shows `[WARN]`, investigate before deciding.

---

## Phase 3: Final Manual Verification (4 min)

### 3.1 Verify the 16-pin connector post-test

After the stress test, power off and physically re-inspect the 16-pin
connector. Look for any new discolouration or heat marks that were not
present before the test.

**WALK AWAY** if the connector shows any signs of heating during the
stress test.

### 3.2 Verify the card identity matches the listing

Confirm that:
- The GPU name shown by `nvidia-smi` matches what the seller advertised
- The serial number matches what was on the listing
- The VBIOS version is in the expected range (98.02.6x.x)

### 3.3 Verify no thermal issues

Check the protocol for:
- GPU temperature stayed below 85C during stress
- No thermal throttling occurred
- Temperature returned to normal after cooldown

### 3.4 Verify the performance baseline

If a 5090 baseline was recorded, confirm the Max-Q's llama-bench results
(pp512 and tg128) are within 10-15% of the 5090. Both cards have identical
1792 GB/s memory bandwidth, so LLM decode speed should be comparable.

### 3.5 Make the decision

Review all results. If everything passes:
- All physical checks passed
- All automated checks passed
- No Xid errors, no ECC errors, no AER errors
- No thermal issues
- Performance is as expected
- 16-pin connector is clean post-test

**Then buy the card.**

If anything is even slightly off, walk away. The cost of a bad card is the
full purchase price. The cost of walking away from a good card is waiting for
the next one.

---

## Walk-Away Conditions (Quick Reference)

### Instant walk-away (physical)

1. Scorched, darkened, or damaged 16-pin power connector
2. Burnt smell
3. Corrosion or liquid damage on PCB
4. Bent or damaged PCIe edge fingers
5. Missing, stripped, or mismatched screws; torn warranty stickers
6. Fans that grind, click, or do not spin freely
7. Damaged DisplayPort connectors
8. Missing or defaced serial number
9. PCB warping or discolouration near VRM
10. Flux residue indicating prior repair

### Instant walk-away (automated, script will catch these)

1. GPU name does not match expected
2. VRAM is not 96 GB
3. VBIOS version does not start with 98.02.6
4. `power.default_limit` is not 300 W (VBIOS tampering)
5. `power.max_limit` exceeds 300 W (shunt mod or flashed VBIOS)
6. PCIe link does not negotiate Gen5 x16 (under load)
7. Any non-zero ECC error at any point
8. Any remapped rows or retired pages (prior contained faults)
9. Any Xid error in kernel log
10. Any AER non-fatal or fatal error
11. gpu-burn fails, crashes, or reports errors
12. gpu-fryer fails, crashes, or detects throttling
13. Power draw below 85% of TDP during gpu-fryer
14. VRAM does not fill to at least 80% during gpu-fryer
15. GPU temp exceeds 90C during stress
16. Memory temp exceeds 100C during stress (if measurable)
17. HW Thermal Slowdown active during stress
18. HW Power Brake active during stress
19. SW Thermal Slowdown active during stress
20. SM clock below expected minimum under load
21. T.Limit specification at zero (possible VBIOS corruption)
22. Any CUDA error during LLM inference
23. LLM output is incoherent or garbage
24. llama-bench pp512/tg128 more than 15% below 5090 baseline
25. VRAM usage does not reach expected levels during model load
26. Temperature does not return to idle after load stops
27. 16-pin connector shows new heat marks after stress test

### If in doubt, walk away.

---

## Config File Reference

The card configuration is in `configs/rtx-pro-6000-maxq.toml`. See the file
for exact values. Key sections:

- **`[card]`**: Card identity (name, VRAM, device ID, subvendor, VBIOS prefix,
  power limits, PCIe gen/width, ECC, min clock speeds under load). Set for the
  Dell OEM variant of the Max-Q (device ID 2BB4, subvendor Dell/1028, VBIOS
  98.02.6x). Do not change.
- **`[thresholds]`**: Temperature and power limits for pass/fail (GPU/mem max
  and walkaway temps, power sustain minimum, idle temp/power, cooldown time
  and temp). Adjust if needed.
- **`[stress]`**: Stress test durations. `gpuburn_duration_s` (default 300s / 5 min
  for ALU error detection) and `gpufryer_duration_s` (default 600s / 10 min for
  full TGP power/thermal and VRAM fill).
- **`[llm.smoke]` and `[llm.fill]`**: Model files and expected VRAM usage.
  `model_file` can be an absolute path or a filename relative to `models_dir`.
- **`[paths]`**: Machine-specific binary paths and models directory. Change
  these for each machine. Can also be overridden via environment variables:

| Variable | Purpose |
|---|---|
| `GPU_CHECK_MODELS_DIR` | Directory containing GGUF model files |
| `GPU_CHECK_GPU_BURN` | Path to gpu_burn binary |
| `GPU_CHECK_GPU_FRYER` | Path to gpu-fryer binary |
| `GPU_CHECK_LLAMA_SERVER` | Path to llama-server binary |
| `GPU_CHECK_LLAMA_BENCH` | Path to llama-bench binary |

---

## Using the Tool on Other Cards

The tool is designed to be configurable. To use it on a different GPU:

1. Create a new TOML config in `configs/` (copy an existing one and modify)
2. Set the expected values for the card (name, VRAM, device ID, power limits,
   PCIe gen/width, ECC support)
3. Set the threshold values (max temps, idle power, etc.)
4. Set the stress test duration
5. Set the LLM model files and expected VRAM usage
6. Set the `[paths]` section for your machine

### Testing the tool

To verify the tool works on your system without running the full stress test:

```bash
python3 gpu-check.py --config configs/rtx-4070-laptop.toml --continue-on-fail --skip-stress --skip-llm --skip-bug-report
```

This runs all the quick checks (identity, power, PCIe, ECC, AER, thermals,
kernel log) without the time-consuming parts.
