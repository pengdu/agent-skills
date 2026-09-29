// Fill-in template for Steps 2 and 5: ablation to attribute the bottleneck,
// then A/B of candidates under rotating buffers with bit-exactness checks.
//
//   cp ablation_template.cu /tmp/probe.cu
//   # edit the four marked sections
//   nvcc -O3 -arch=sm_75 -I<skill>/scripts /tmp/probe.cu -o /tmp/probe && /tmp/probe
//
// The structure is what matters, not the example kernel: identical launch
// geometry and identical indexing across variants, so the delta between two
// variants isolates exactly one stage. The example below is an
// uint8 NHWC -> float NCHW conversion; replace it with the kernel under study.

#include "ab_harness.cuh"

#include <cstdint>

// ===========================================================================
// SECTION 1 - the kernel under study, parameterised by what work it does.
//
// kFull      the real kernel
// kReadOnly  reads, accumulates, guards the store with a data-dependent
//            condition that is never true. The loads survive dead-code
//            elimination; no store traffic is generated.
// kWriteOnly stores a constant; no loads.
// kNoMath    reads and stores, skipping the arithmetic.
// ===========================================================================

enum Mode { kFull, kReadOnly, kWriteOnly, kNoMath };

struct Scale {
    float mean[3];
    float stddev[3];
};

template <int C, Mode M>
__global__ void Kernel(const uint8_t* __restrict__ src, int src_step, float* __restrict__ dst,
                       int dst_step_floats, int w, int h, Scale s, float* sink) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const uint8_t* sp = src + (size_t)y * src_step + (size_t)x * C;
    const size_t plane = (size_t)h * dst_step_floats;
    float* dp = dst + (size_t)y * dst_step_floats + x;

    if (M == kWriteOnly) {
#pragma unroll
        for (int ch = 0; ch < C; ++ch) dp[ch * plane] = 0.5f;
        return;
    }

    float acc = 0.0f;
#pragma unroll
    for (int ch = 0; ch < C; ++ch) {
        float v = (float)sp[ch];
        if (M == kFull) {
            v = v / 255.0f;
            v = (v - s.mean[ch]) / s.stddev[ch];
        }
        if (M == kReadOnly) acc += v;
        else dp[ch * plane] = v;
    }
    if (M == kReadOnly && acc == 1.2345678e30f) sink[0] = acc;
}

// ===========================================================================
// SECTION 2 - candidate variants to A/B against the baseline.
// Add one template parameter per axis you want to sweep; keep the arithmetic
// sequence identical if the output is contractually pinned.
//
// The example candidate below is deliberately naive: N pixels per thread with
// only the scalar tail path and no vectorised fast path, so it measures ~0.65x
// out of the box. That is the point of the template -- it shows what a losing
// candidate looks like in the output table. Replace it with a real candidate.
// ===========================================================================

template <int C, int N>
__global__ void Candidate(const uint8_t* __restrict__ src, int src_step, float* __restrict__ dst,
                          int dst_step_floats, int w, int h, Scale s) {
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x0 >= w || y >= h) return;
    const uint8_t* sp = src + (size_t)y * src_step + (size_t)x0 * C;
    const size_t plane = (size_t)h * dst_step_floats;
    float* dp = dst + (size_t)y * dst_step_floats + x0;

    // Ragged tail handled scalar-wise, so odd widths stay correct. Always keep
    // a tail path and always include an odd width in the shape list.
    for (int i = 0; i < N && x0 + i < w; ++i) {
#pragma unroll
        for (int ch = 0; ch < C; ++ch) {
            float v = (float)sp[i * C + ch];
            v = v / 255.0f;
            v = (v - s.mean[ch]) / s.stddev[ch];
            dp[i + ch * plane] = v;
        }
    }
}

// ===========================================================================
// SECTION 3 - shapes. Include the shapes production actually uses, plus one
// odd width to exercise tail paths, plus one shape whose working set fits in L2
// and one that does not.
// ===========================================================================

struct Shape {
    int w, h, c;
    const char* note;
};

static const Shape kShapes[] = {
    {360, 360, 3, "small"},
    {512, 512, 3, "fits_L2"},
    {1280, 720, 3, "mid"},
    {1920, 1080, 3, "large"},
    {1921, 1080, 3, "odd_width"},
};

// ===========================================================================
// SECTION 4 - drivers. Usually no edits needed below here.
// ===========================================================================

int warmup = 20, iters = 200, repeats = 3;

template <int C>
void Ablate(const Shape& sh, const abh::DeviceInfo& d) {
    Scale s{{0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f}};
    abh::RotatingSet bufs((size_t)sh.w * C, sh.h, (size_t)sh.w * sizeof(float),
                          (size_t)sh.h * C, d);
    float* sink = nullptr;
    ABH_CK(cudaMalloc(&sink, sizeof(float)));

    const int ss = bufs.src_pitch();
    const int ds = bufs.dst_pitch() / (int)sizeof(float);
    const dim3 blk(32, 8);
    const dim3 grid((sh.w + blk.x - 1) / blk.x, (sh.h + blk.y - 1) / blk.y);

    auto run = [&](auto kernel) {
        return abh::TimeBest(
            [&](int it) {
                kernel<<<grid, blk>>>((const uint8_t*)bufs.src(it), ss, (float*)bufs.dst(it), ds,
                                      sh.w, sh.h, s, sink);
            },
            warmup, iters, repeats);
    };
    const abh::Timing ro = run(Kernel<C, kReadOnly>);
    const abh::Timing wo = run(Kernel<C, kWriteOnly>);
    const abh::Timing nm = run(Kernel<C, kNoMath>);
    const abh::Timing fu = run(Kernel<C, kFull>);
    ABH_CK(cudaGetLastError());

    const double px = (double)sh.w * sh.h;
    const double rd = px * C;
    const double wr = px * C * sizeof(float);
    std::printf("%-12s %4dx%-5d c%d | %8.4f %8.4f %8.4f %8.4f | %7.1f %7.1f %7.1f %7.1f\n",
                sh.note, sh.w, sh.h, C, ro.min_ms, wo.min_ms, nm.min_ms, fu.min_ms,
                abh::GBs(rd, ro.min_ms), abh::GBs(wr, wo.min_ms), abh::GBs(rd + wr, nm.min_ms),
                abh::GBs(rd + wr, fu.min_ms));
    ABH_CK(cudaFree(sink));
}

template <int C>
void Compare(const Shape& sh, const abh::DeviceInfo& d) {
    Scale s{{0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f}};
    abh::RotatingSet bufs((size_t)sh.w * C, sh.h, (size_t)sh.w * sizeof(float),
                          (size_t)sh.h * C, d);
    float* ref = nullptr;
    ABH_CK(cudaMalloc(&ref, bufs.dst_bytes()));

    const int ss = bufs.src_pitch();
    const int ds = bufs.dst_pitch() / (int)sizeof(float);
    const dim3 blk(32, 8);
    const dim3 g1((sh.w + blk.x - 1) / blk.x, (sh.h + blk.y - 1) / blk.y);
    const int w4 = (sh.w + 3) / 4;
    const dim3 g4((w4 + blk.x - 1) / blk.x, (sh.h + blk.y - 1) / blk.y);

    // Bit-exactness first: a faster wrong candidate is not a candidate.
    ABH_CK(cudaMemset(ref, 0xA5, bufs.dst_bytes()));
    ABH_CK(cudaMemset(bufs.dst0(), 0x5A, bufs.dst_bytes()));
    Kernel<C, kFull><<<g1, blk>>>((const uint8_t*)bufs.src0(), ss, ref, ds, sh.w, sh.h, s, nullptr);
    Candidate<C, 4><<<g4, blk>>>((const uint8_t*)bufs.src0(), ss, (float*)bufs.dst0(), ds, sh.w,
                                 sh.h, s);
    ABH_CK(cudaDeviceSynchronize());
    const bool exact =
        abh::BitExactDevice(ref, bufs.dst0(), bufs.dst_bytes(), (size_t)sh.h * C,
                            (size_t)sh.w * sizeof(float), (size_t)bufs.dst_pitch());

    // Rotating: what production will see.
    const abh::Timing base_rot = abh::TimeBest(
        [&](int it) {
            Kernel<C, kFull><<<g1, blk>>>((const uint8_t*)bufs.src(it), ss, (float*)bufs.dst(it),
                                          ds, sh.w, sh.h, s, nullptr);
        },
        warmup, iters, repeats);
    const abh::Timing cand_rot = abh::TimeBest(
        [&](int it) {
            Candidate<C, 4><<<g4, blk>>>((const uint8_t*)bufs.src(it), ss, (float*)bufs.dst(it), ds,
                                         sh.w, sh.h, s);
        },
        warmup, iters, repeats);

    // Reuse: the biased measurement, reported on purpose so a gain that only
    // exists here is visibly an artefact.
    const abh::Timing base_reuse = abh::TimeBest(
        [&](int) {
            Kernel<C, kFull><<<g1, blk>>>((const uint8_t*)bufs.src0(), ss, (float*)bufs.dst0(), ds,
                                          sh.w, sh.h, s, nullptr);
        },
        warmup, iters, repeats);
    const abh::Timing cand_reuse = abh::TimeBest(
        [&](int) {
            Candidate<C, 4><<<g4, blk>>>((const uint8_t*)bufs.src0(), ss, (float*)bufs.dst0(), ds,
                                         sh.w, sh.h, s);
        },
        warmup, iters, repeats);
    ABH_CK(cudaGetLastError());

    std::printf("%-12s %4dx%-5d c%d slots=%-2d | %8.4f %8.4f %6.2fx | %8.4f %8.4f %6.2fx | %s\n",
                sh.note, sh.w, sh.h, C, bufs.slots(), base_rot.min_ms, cand_rot.min_ms,
                base_rot.min_ms / cand_rot.min_ms, base_reuse.min_ms, cand_reuse.min_ms,
                base_reuse.min_ms / cand_reuse.min_ms, exact ? "bit-exact" : "DIFFERS");
    ABH_CK(cudaFree(ref));
}

int main(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--warmup")) warmup = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--repeats")) repeats = std::atoi(argv[i + 1]);
    }
    const abh::DeviceInfo d = abh::Query();
    abh::PrintDevice(d);
    std::printf("run bandwidth_probe first; use its copy number as the denominator\n\n");

    std::printf("== Step 4: occupancy\n");
    abh::ReportOccupancy("Kernel<3,kFull>", Kernel<3, kFull>, 256, d);
    abh::ReportOccupancy("Candidate<3,4>", Candidate<3, 4>, 256, d);
    std::printf("\n");

    std::printf("== Step 2: ablation (min_ms | GB/s of that stage's own traffic)\n");
    std::printf("%-12s %-13s | %8s %8s %8s %8s | %7s %7s %7s %7s\n", "shape", "", "rd_only",
                "wr_only", "no_math", "full", "rd_GBs", "wr_GBs", "nm_GBs", "full_GBs");
    for (const Shape& s : kShapes) Ablate<3>(s, d);
    std::printf("\n");

    std::printf("== Step 5: candidate A/B (rotating = real; reuse = biased, for contrast)\n");
    std::printf("%-12s %-16s | %8s %8s %6s | %8s %8s %6s | %s\n", "shape", "", "base_rot",
                "cand_rot", "gain", "base_ru", "cand_ru", "gain", "correctness");
    for (const Shape& s : kShapes) Compare<3>(s, d);
    std::printf("\n");

    std::printf("== launch floor (check before optimizing a small-shape kernel body)\n");
    abh::ReportLaunchFloor(dim3(60, 135), dim3(32, 8), warmup, iters);
    return 0;
}
