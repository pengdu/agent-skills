# Worked example: fused uint8 NHWC -> float NCHW on a 2080 Ti

The case this skill was distilled from. Read it for the shape of the reasoning,
particularly how the promising hypothesis died.

Constraints: RTX 2080 Ti (SM_75), driver 470.57.02, inside a container without
`CAP_SYS_ADMIN`. `ncu` unusable (`ERR_NVGPUCTRPERM`), `nvprof` unsupported on
SM 7.5+. Kernel contract: output must be bit-identical to the NPP pipeline it
replaced, so `/255` and `/stddev` must stay IEEE divisions.

## Step 1 - roofline denominator

```
read 581   write 564   copy 527 GB/s      (spec sheet: 616)
```

## Step 2 - ablation, 1920x1080 c3

| stage | min_ms | GB/s of its own traffic | vs achievable |
|---|---|---|---|
| read only | 0.0188 | 330 | 57% |
| write only | 0.0456 | 545 | 97% |
| read+write, no math | 0.0676 | 460 | — |
| full | 0.0652 | 477 | 90% of copy |

Reading: writes are done, leave them. `full ≈ no_math` so the divisions are
free. Read is at 57% -> hypothesis, the read path has structural waste.

## Step 3 - static evidence

```
16x LDG.E.U8.CONSTANT.SYS     3 single-byte gathers per thread (C=3)
16x STG.E.SYS                 4-byte scalar stores
18-19 registers, 0 spill
```

Consistent with the hypothesis: a warp reads 32 bytes at stride 3 spanning
96 bytes, as three separate gather instructions.

## Step 4 - occupancy

100% for every variant, 19-31 registers, no spills, on a 64K register file.
Occupancy is not a factor; do not use it to explain anything here.

## Step 5 - candidates, and the refutation

4 pixels per thread, 3x 32-bit loads, 3x 128-bit stores. Bit-exact. Result:

| shape | baseline | 4px | |
|---|---|---|---|
| 1920x1080 c3 | 0.0637 | 0.0668 | **0.95x** |
| 1280x720 c3 | 0.0300 | 0.0313 | 0.96x |
| 512x512 c1 | 0.0053 | 0.0046 | 1.17x |
| 1920x1080 c1 | 0.0276 | 0.0240 | 1.15x |

**Hypothesis refuted for C=3.** Not registers (100% occupancy either way), not
correctness. The byte gathers were never costing DRAM traffic; L1 absorbed the
redundancy. The `330 GB/s` read figure was real but was not the limiter, because
the full kernel was already at 93% of achievable copy bandwidth.

Then two more axes, both negative:

- block geometry, 7 configurations from 32x8 to 256x1: all within ±1%
- `__stcs` / `__ldcs` streaming hints: 1.37x under buffer reuse, **1.00x under
  rotation**. Artefact (see traps #2)

## The real finding

Small shapes did not scale with size and every candidate plateaued at the same
0.0067-0.0076 ms. That plateau is per-launch cost. The production caller was:

```cpp
for (int i = 0; i < batch; i++)
    CudaUint8NHWCToFloatNCHW(src + i*srcFrame, ..., dst + i*dstFrame, ...);
```

Moving the frame index into `grid.z`, one launch for the whole batch, kernel body
otherwise untouched:

| shape | loop | batched | |
|---|---|---|---|
| 512x512 c1 n8 | 0.0406 | 0.0276 | **1.47x** |
| 360x360 c3 n8 | 0.0472 | 0.0353 | **1.34x** |
| 384x384 c3 n8 | 0.0512 | 0.0380 | 1.35x |
| 540x960 c3 n8 | 0.1491 | 0.1434 | 1.04x |
| 1920x1080 c3 n1 | 0.0658 | 0.0658 | 1.00x |

Bit-identical, no regression anywhere, and the large shapes are unchanged for a
reason the evidence already explained: they were at 93% of achievable.

## Conclusion block as delivered

| evidence | number | licenses |
|---|---|---|
| bandwidth probe | copy 527 GB/s, not 616 | correct denominator |
| write-only ablation | 545 GB/s = 97% | writes are done |
| read-only ablation + SASS | 330 GB/s, 3x LDG.U8 | hypothesis only |
| 4px candidate A/B | C=3 0.95x, occupancy 100% | **hypothesis refuted** |
| full vs roofline | 488/527 = 93% at 1080p | no structural headroom |
| rotating vs reuse | `__stcs` 1.37x -> 1.00x | **artefact, discarded** |
| launch floor | small shapes plateau | bottleneck is at the call site |
| batched launch A/B | 1.34-1.47x at batch 8 | **the only change to make** |

Net: no kernel algorithm change. One call-site change. Three hypotheses killed
by measurement, each documented so nobody retries them.
