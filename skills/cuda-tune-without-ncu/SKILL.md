---
name: cuda-tune-without-ncu
description: "Evidence-driven CUDA kernel optimization when Nsight Compute is unavailable (no CAP_SYS_ADMIN in container, ERR_NVGPUCTRPERM, driver too old for the installed ncu, nvprof unsupported on SM 7.5+). Replaces NCU counters with four measurable evidence sources: achievable-bandwidth microbenchmarks (roofline denominator), read/write/no-math ablation (bottleneck attribution), SASS + ptxas static inspection (coalescing, instruction mix, registers, spills), and the occupancy API (occupancy limiter). Enforces rotating-buffer A/B benchmarking so L2-reuse artefacts cannot be mistaken for speedups, and requires every hypothesis to be confirmed or refuted by measurement before any kernel edit. Use when asked to optimize, profile, or speed up CUDA kernels and ncu cannot collect counters."
---

# Optimizing CUDA kernels without Nsight Compute

## When this applies

Use `cuda-auto-tune` if `ncu` can actually collect counters. Use this skill only after
confirming it cannot. Confirm, do not assume:

```bash
ncu --version                                   # exists?
ncu --metrics sm__cycles_elapsed.avg <trivial>  # can it attach and collect?
```

Known dead ends, do not spend time on them:

| Symptom | Cause | Fix available here? |
|---|---|---|
| `ERR_NVGPUCTRPERM`, `RmProfilingAdminOnly: 1` | container lacks `CAP_SYS_ADMIN` | No. Needs `--cap-add=SYS_ADMIN` or host `NVreg_RestrictProfilingToAdminUsers=0` |
| ncu injection refuses to connect to driver | ncu newer than driver (2023.x needs r525+, 2026.x needs r580+) | Sometimes: install the ncu matching the CUDA toolkit (`nsight-compute-11-4` for CUDA 11.4 / r470) |
| `nvprof` refuses to run | unsupported on compute capability 7.5+ | No. Dead end |
| Nsight Systems available | timeline only, no L1/L2 hit rate, warp stalls, coalescing, bank conflicts | Partially: gives SM active%, DRAM GB/s, kernel duration |

## GATE CHECK

```
STOP — Do you have measured evidence for this specific kernel?
  NO  → collect it (Steps 1-4). Do NOT touch kernel code.
  YES → is your bottleneck claim attributable to a number you measured,
        not to a mechanism you inferred?
          NO  → keep measuring.
          YES → proceed to Step 5.
```

Two rules that carry over from `cuda-auto-tune`, unchanged:

- **No data, no edit.** Reading SASS and forming a hypothesis is not data. A hypothesis
  becomes data only once an ablation or A/B measures it.
- **Confidence must be downgraded explicitly.** Without NCU you cannot say "L2 hit rate
  is X, therefore Y". You can say "read-only ablation runs at 57% of measured achievable
  read bandwidth, so the read path has structural waste" — and then you must test the fix,
  because the inferred cause is frequently wrong. In practice most SASS-derived hypotheses
  get refuted at this step; that is the skill working, not failing.

## The four evidence sources

| Source | NCU dimension it replaces | Tool |
|---|---|---|
| Achievable-bandwidth microbenchmark | roofline denominator | `scripts/bandwidth_probe.cu` |
| Read / write / no-math / full ablation | bottleneck attribution, instruction vs memory bound | `scripts/ab_harness.cuh` |
| `cuobjdump -sass`, `ptxas -v` | coalescing ratio, instruction mix, registers, spills, smem | `scripts/static_evidence.sh` |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | achieved occupancy + limiter | `abh::ReportOccupancy` |

Not obtainable: L1/L2 hit rate, warp stall reasons, bank conflicts, sector counts. Never
claim these. Where they matter, substitute a working-set sweep and say so.

## Workflow

### Step 1 — Establish the roofline denominator

```bash
nvcc -O3 -arch=sm_XX scripts/bandwidth_probe.cu -o bw && ./bw
```

Record read / write / copy GB/s. **Use the measured copy number as the denominator for
read+write kernels, never the spec sheet.** On a 2080 Ti the spec sheet says 616 GB/s
while copy measures 527; using 616 understates achieved efficiency by 17% and invents
headroom that does not exist.

### Step 2 — Attribute the bottleneck by ablation

Build variants of the *same* kernel with identical launch geometry and indexing, changing
only what work happens: read-only (accumulate, guard the store with a data-dependent
condition that is never true, so loads are not eliminated), write-only (store a constant),
read+write with no arithmetic, and full. Compute each variant's GB/s against the traffic
*that variant* needs.

Read this as:
- a stage at >90% of its achievable bandwidth is done, leave it alone
- `full ≈ no_math` means arithmetic is free, even expensive IEEE divisions
- `full ≈ read_only + write_only` means no overlap; the kernel is bandwidth-serialised
- a stage far below achievable *and* the kernel already near the roofline means the stage
  is not what is limiting the kernel

### Step 3 — Static evidence

```bash
bash scripts/static_evidence.sh path/to/kernel.cu sm_75 [extra nvcc flags]
```

Reports registers/spills/smem per kernel and a histogram of load/store widths. `LDG.E.U8`
means single-byte scalar loads; `STG.E.128` means 128-bit stores. This is where hypotheses
come from — not where they are confirmed.

### Step 4 — Rule out occupancy

`abh::ReportOccupancy` gives registers, blocks/SM, warps/SM, and theoretical occupancy.
Check this *before* blaming register pressure for anything. On Turing's 64K register file
a kernel can grow from 19 to 31 registers and still hold 100% occupancy, so "the
vectorised version used more registers" is usually not the explanation.

### Step 5 — A/B candidates, with the artefact controls on

**Mandatory: rotating buffers.** A harness that hands the kernel the same buffers every
iteration measures L2 reuse that production will never see. `abh::RotatingSet` allocates
enough identically-pitched buffer pairs to cover several times L2 and cycles them.

Measure every candidate *both* ways and report both. A gain that exists under reuse and
vanishes under rotation is an artefact — discard it. This single control eliminated a
1.37x "win" in the case this skill was distilled from.

**Mandatory: repeats.** Use `abh::TimeBest(..., repeats=3)`. A single shape behaving
differently from its neighbours is noise until it survives repeats.

**Mandatory: bit-exactness.** If the kernel's contract pins its output (matching a library
path, a reference implementation, a golden file), compare every output byte with
`abh::BitExactRows`, not a tolerance. Report correctness alongside timing in the same table
so a faster-but-wrong candidate cannot be read as a win.

### Step 6 — Conclusion block, then edit

Before editing, write out a table with one row per piece of evidence: the measurement, the
number, and what it licenses you to conclude. Include the refuted hypotheses — they are
the most valuable rows, because they are what stops the next person redoing the work.
Then change only what the table supports.

After editing: rerun the project's correctness tests and the A/B. Put the measured
before/after numbers in a comment next to the change, with the shapes and the harness
conditions, so the next reader can tell whether the number still applies to their case.

## Traps

See `references/traps.md` for the full list with numbers. The ones that cost the most time:

- **Spec-sheet bandwidth.** Always measure.
- **Buffer reuse in the harness.** See Step 5.
- **Launch-overhead floor.** Small shapes plateau at a per-launch cost and stop scaling
  with size. When a kernel does not scale down, the fix is usually outside the kernel —
  fewer launches (batch via `grid.z`, CUDA graphs, merged calls) rather than a faster
  kernel. Check the call site before optimizing the kernel body.
- **Believing the SASS.** Byte-wise gathers look catastrophic and often cost nothing,
  because L1 absorbs the redundancy and DRAM traffic is unchanged. Vectorising them can be
  a regression.
- **Reporting mean only.** Report min and mean; a gap between them is interference, and
  min is the more stable comparator for A/B.

## Files

| Path | Use |
|---|---|
| `scripts/bandwidth_probe.cu` | Step 1. Standalone, no edits needed |
| `scripts/static_evidence.sh` | Step 3. `static_evidence.sh <file.cu> <arch> [nvcc flags]` |
| `scripts/ab_harness.cuh` | Steps 4-5. Header-only: `TimeBest`, `RotatingSet`, `BitExactDevice`, `ReportOccupancy`, `ReportLaunchFloor` |
| `scripts/ablation_template.cu` | Steps 2+5 driver. Copy, edit the four marked sections |
| `references/traps.md` | The artefacts and dead ends, with the numbers that exposed them |
| `references/worked-example.md` | A full run where the leading hypothesis was refuted and the fix turned out to be at the call site |
