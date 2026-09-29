# Traps

Each of these cost real time in the case this skill was distilled from (fused
uint8 NHWC -> float NCHW conversion on a 2080 Ti, SM_75, driver 470).

## 1. Spec-sheet bandwidth

`2 * memoryClockRate * memoryBusWidth / 8` is unreachable. Measured on the same
card:

| | GB/s | % of spec |
|---|---|---|
| spec sheet | 616 | 100% |
| float4 read | 581 | 94% |
| float4 write | 564 | 92% |
| float4 copy (read+write) | 527 | 86% |

A kernel measured at 488 GB/s is at 79% of spec but **93% of achievable**. The
first number invites a rewrite; the second correctly says stop. Always run
`bandwidth_probe.cu` first and quote the copy figure for read+write kernels.

## 2. Buffer reuse in the harness

The single most dangerous artefact. A harness that passes the same buffers on
every iteration lets iteration N hit lines iteration N-1 left in L2.
Cache-bypass hints then look like large wins:

| shape | `__stcs` gain, reuse | `__stcs` gain, rotating |
|---|---|---|
| 1920x1080 c1 | **1.37x** | 1.00x |
| 540x960 c3 | **1.16x** | 1.01x |

The mechanism: not write-allocating the destination leaves the *source* resident
across iterations, so the read becomes nearly free. Production hands the kernel
a new frame each call and never gets this. `abh::RotatingSet` fixes it; report
both columns so the artefact is visible rather than hidden.

Related: a shape whose entire working set fits in L2 can measure *above* DRAM
peak. If a row shows more GB/s than `bandwidth_probe` reported, its working set
is cache-resident and its number is not a DRAM measurement.

## 3. One shape disagreeing with its neighbours

A 1.15x that appeared only at 512x512, reproduced 5/5 within one harness, then
vanished when re-measured in a differently-structured harness. Allocation order
changes addresses, which changes L2 set conflicts. Rule: a gain must survive
`repeats >= 3` **and** be consistent with its neighbouring shapes, or be
re-measured in a second harness before it is believed.

## 4. The launch-overhead floor

Small shapes stop scaling with size. A kernel needing 1.94 MB of traffic
"should" take 0.0037 ms at 527 GB/s but measured 0.0075 ms, and every block
geometry and every candidate landed on the same 0.0067-0.0076 ms plateau.

That plateau is the launch cost, not the kernel. No kernel edit moves it. Look
at the call site instead:

- a per-frame or per-tile host loop -> collapse into one launch via `grid.z`
- many small independent launches -> CUDA graphs
- repeated tiny kernels in sequence -> fuse

In the source case, replacing a per-frame host loop with `grid.z` gave 1.36x at
360x360 batch 8 and 1.47x at 512x512 c1 batch 8, while the kernel body was left
untouched. **Check the caller before optimizing the kernel.**

## 5. Believing the SASS

`cuobjdump` showed 3x `LDG.E.U8` per thread for a 3-channel interleaved read:
three single-byte gathers, warp addresses spanning 96 bytes at stride 3. It
looks like the obvious defect, and the read-only ablation agreed (330 GB/s
against 581 achievable).

Vectorising it -- 4 pixels per thread, 3x 32-bit loads, 3x 128-bit stores --
made C=3 **5% slower**, at 100% occupancy in both versions and no spills. The
byte gathers cost nothing at DRAM level because L1 absorbs the redundancy; the
DRAM traffic was identical, and the wider version just had fewer warps covering
more x-range each.

The same change gave +15% for C=1. Channel count flipped the sign. SASS
generates hypotheses; only the A/B decides.

## 6. Dead ends that look installable

- `nvprof` is refused on compute capability 7.5 and above. No flag re-enables it.
- ncu newer than the driver fails in its injection library, not with a clear
  version error. 2023.x needs r525+, 2026.x needs r580+. Match ncu to the CUDA
  toolkit: `nsight-compute-11-4` for CUDA 11.4 / driver 470.
- `ERR_NVGPUCTRPERM` with `RmProfilingAdminOnly: 1` in
  `/proc/driver/nvidia/params` is a capability problem, not a version problem.
  Being root in a container is not enough; `CAP_SYS_ADMIN` must be granted at
  container start. Nothing inside the container fixes it.
- Nsight Systems is not a substitute. It gives SM active%, DRAM GB/s and kernel
  duration, but not L1/L2 hit rate, warp stall reasons, coalescing ratio or bank
  conflicts.
- CUPTI's `profiling_injection` sample sits on the same PerfWorks layer ncu uses
  and hits the same `RmProfilingAdminOnly` gate. Not a way around it.

## 7. Arithmetic that must not be "optimized"

If a kernel's contract is bit-exact agreement with another implementation, the
usual arithmetic strength reductions are forbidden even though they are faster:
`__fdividef`, multiplying by a precomputed reciprocal, and fast-math flags all
drift in the last mantissa bits.

Check whether it even matters before arguing about it: the ablation showed
`full ≈ no_math` for C=3, i.e. two IEEE divisions per channel cost nothing
measurable because the kernel is memory-bound. Compare with a bit-exactness
check, not a tolerance -- a tolerance hides exactly this drift.
