// Header-only A/B harness for kernel optimization without NCU.
//
//   #include "ab_harness.cuh"
//
// Provides the four controls that stop a measurement from lying:
//
//   abh::TimeBest      min over repeated timing runs, not one run
//   abh::RotatingSet   cycles distinct buffer pairs covering several times L2,
//                      so no iteration reuses another's cache lines
//   abh::BitExactRows  byte comparison of pitched output, not a tolerance
//   abh::ReportOccupancy  registers / blocks per SM / theoretical occupancy
//
// Everything reports both mean and min. A wide mean-min gap means interference;
// min is the more stable comparator when ranking candidates.

#ifndef ABH_AB_HARNESS_CUH_
#define ABH_AB_HARNESS_CUH_

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace abh {

#define ABH_CK(call)                                                                \
    do {                                                                            \
        cudaError_t e_ = (call);                                                     \
        if (e_ != cudaSuccess) {                                                     \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,      \
                         cudaGetErrorString(e_));                                    \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

struct Timing {
    double mean_ms = 0.0;
    double min_ms = 0.0;
};

// fn takes the iteration index so callers can rotate buffers per iteration.
template <typename Fn>
Timing TimeIt(Fn&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn(i);
    ABH_CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    ABH_CK(cudaEventCreate(&a));
    ABH_CK(cudaEventCreate(&b));
    std::vector<float> s(iters);
    for (int i = 0; i < iters; ++i) {
        ABH_CK(cudaEventRecord(a));
        fn(i);
        ABH_CK(cudaEventRecord(b));
        ABH_CK(cudaEventSynchronize(b));
        ABH_CK(cudaEventElapsedTime(&s[i], a, b));
    }
    ABH_CK(cudaEventDestroy(a));
    ABH_CK(cudaEventDestroy(b));
    double sum = 0.0;
    for (float v : s) sum += v;
    return Timing{sum / iters, *std::min_element(s.begin(), s.end())};
}

// Repeat the whole timing run and keep the best. A candidate that only wins in
// one repeat did not win.
template <typename Fn>
Timing TimeBest(Fn&& fn, int warmup, int iters, int repeats = 3) {
    Timing best{1e30, 1e30};
    for (int r = 0; r < repeats; ++r) {
        const Timing t = TimeIt(fn, warmup, iters);
        if (t.min_ms < best.min_ms) best.min_ms = t.min_ms;
        if (t.mean_ms < best.mean_ms) best.mean_ms = t.mean_ms;
    }
    return best;
}

inline double GBs(double bytes, double ms) { return bytes / (ms * 1e-3) / 1e9; }

// ---------------------------------------------------------------- device info

struct DeviceInfo {
    cudaDeviceProp prop{};
    int sms = 0;
    int max_threads_per_sm = 0;
    size_t l2_bytes = 0;
    double spec_gbs = 0.0;
};

inline DeviceInfo Query(int device = 0) {
    DeviceInfo d;
    ABH_CK(cudaGetDeviceProperties(&d.prop, device));
    ABH_CK(cudaSetDevice(device));
    ABH_CK(cudaFree(nullptr));  // force context creation
    d.sms = d.prop.multiProcessorCount;
    d.max_threads_per_sm = d.prop.maxThreadsPerMultiProcessor;
    d.l2_bytes = (size_t)d.prop.l2CacheSize;
    d.spec_gbs = 2.0 * d.prop.memoryClockRate * 1e3 * (d.prop.memoryBusWidth / 8) / 1e9;
    return d;
}

inline void PrintDevice(const DeviceInfo& d) {
    std::printf("device: %s  SM%d%d  SMs=%d  L2=%.2f MiB  spec=%.1f GB/s (not achievable)\n",
                d.prop.name, d.prop.major, d.prop.minor, d.sms, d.l2_bytes / 1048576.0,
                d.spec_gbs);
}

// ----------------------------------------------------------------- occupancy

// Check this before blaming register pressure. A kernel can gain a dozen
// registers and keep full occupancy.
template <typename Kernel>
void ReportOccupancy(const char* name, Kernel kernel, int block_threads,
                     const DeviceInfo& d, size_t dynamic_smem = 0) {
    cudaFuncAttributes at{};
    ABH_CK(cudaFuncGetAttributes(&at, (const void*)kernel));
    int blocks = 0;
    ABH_CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (const void*)kernel,
                                                         block_threads, dynamic_smem));
    std::printf("%-28s regs=%-3d smem=%-6zu spill(st/ld)=%zu/%zu blocks/SM=%-3d warps/SM=%-3d "
                "occ=%5.1f%%\n",
                name, at.numRegs, (size_t)at.sharedSizeBytes, (size_t)at.localSizeBytes,
                (size_t)at.localSizeBytes, blocks, blocks * block_threads / 32,
                100.0 * blocks * block_threads / d.max_threads_per_sm);
}

// ------------------------------------------------------------ rotating buffers
//
// THE control that matters most. A harness reusing one buffer pair measures L2
// reuse across iterations that production will never get, which makes
// cache-bypass tricks (__stcs, __ldcs, non-temporal hints) look like large wins
// that evaporate in the real caller.
//
// Allocates `slots` identically-pitched src/dst pairs such that one full cycle
// touches at least `l2_multiple` times L2. All pitches are asserted equal so a
// candidate kernel can be pointed at any slot without changing its arguments.

class RotatingSet {
public:
    RotatingSet(size_t src_width_bytes, size_t src_height, size_t dst_width_bytes,
                size_t dst_height, const DeviceInfo& d, int l2_multiple = 4, int max_slots = 64) {
        // Probe the pitches once.
        void* t = nullptr;
        void* u = nullptr;
        ABH_CK(cudaMallocPitch(&t, &src_pitch_, src_width_bytes, src_height));
        ABH_CK(cudaMallocPitch(&u, &dst_pitch_, dst_width_bytes, dst_height));
        ABH_CK(cudaFree(t));
        ABH_CK(cudaFree(u));
        src_bytes_ = src_pitch_ * src_height;
        dst_bytes_ = dst_pitch_ * dst_height;

        const size_t want = (size_t)l2_multiple * d.l2_bytes;
        const size_t per_slot = src_bytes_ + dst_bytes_;
        slots_ = (int)((want + per_slot - 1) / per_slot);
        slots_ = std::max(2, std::min(slots_, max_slots));

        srcs_.resize(slots_);
        dsts_.resize(slots_);
        for (int i = 0; i < slots_; ++i) {
            size_t sp = 0, dp = 0;
            ABH_CK(cudaMallocPitch(&srcs_[i], &sp, src_width_bytes, src_height));
            ABH_CK(cudaMallocPitch(&dsts_[i], &dp, dst_width_bytes, dst_height));
            if (sp != src_pitch_ || dp != dst_pitch_) {
                std::fprintf(stderr, "RotatingSet: pitch differed between slots\n");
                std::exit(1);
            }
            // Distinct fill per slot so a kernel that ignores the slot pointer
            // produces visibly wrong output instead of matching by accident.
            ABH_CK(cudaMemset(srcs_[i], (int)(0x21 + i * 7), src_bytes_));
        }
    }

    ~RotatingSet() {
        for (void* p : srcs_) cudaFree(p);
        for (void* p : dsts_) cudaFree(p);
    }

    RotatingSet(const RotatingSet&) = delete;
    RotatingSet& operator=(const RotatingSet&) = delete;

    int slots() const { return slots_; }
    int src_pitch() const { return (int)src_pitch_; }
    int dst_pitch() const { return (int)dst_pitch_; }
    size_t src_bytes() const { return src_bytes_; }
    size_t dst_bytes() const { return dst_bytes_; }

    // Rotating access: use these inside a TimeIt lambda.
    void* src(int iter) const { return srcs_[iter % slots_]; }
    void* dst(int iter) const { return dsts_[iter % slots_]; }

    // Fixed access: use these to reproduce the reuse-biased measurement on
    // purpose, so the two can be reported side by side.
    void* src0() const { return srcs_[0]; }
    void* dst0() const { return dsts_[0]; }

private:
    size_t src_pitch_ = 0, dst_pitch_ = 0, src_bytes_ = 0, dst_bytes_ = 0;
    int slots_ = 0;
    std::vector<void*> srcs_, dsts_;
};

// ------------------------------------------------------------- bit-exactness
//
// Use this, not a tolerance, whenever the kernel's contract pins its output
// (matching a library pipeline, a reference kernel, a golden file). A tolerance
// hides exactly the last-mantissa-bit drift that makes a "faster" candidate
// unusable, e.g. replacing an IEEE division with a reciprocal multiply.
//
// Compares only the meaningful part of each row; pitch padding is skipped
// because neither side is required to write it.

inline bool BitExactRows(const void* a, const void* b, size_t rows, size_t row_bytes,
                         size_t pitch) {
    const unsigned char* pa = (const unsigned char*)a;
    const unsigned char* pb = (const unsigned char*)b;
    for (size_t r = 0; r < rows; ++r) {
        if (std::memcmp(pa + r * pitch, pb + r * pitch, row_bytes) != 0) return false;
    }
    return true;
}

// Device-side convenience: copies both buffers back and compares.
inline bool BitExactDevice(const void* d_a, const void* d_b, size_t total_bytes, size_t rows,
                           size_t row_bytes, size_t pitch) {
    std::vector<unsigned char> ha(total_bytes), hb(total_bytes);
    ABH_CK(cudaMemcpy(ha.data(), d_a, total_bytes, cudaMemcpyDeviceToHost));
    ABH_CK(cudaMemcpy(hb.data(), d_b, total_bytes, cudaMemcpyDeviceToHost));
    return BitExactRows(ha.data(), hb.data(), rows, row_bytes, pitch);
}

// ------------------------------------------------------------ launch overhead
//
// Small shapes stop scaling with size once the per-launch cost dominates. When
// that happens no kernel edit helps and the fix is at the call site: batch
// frames into grid.z, merge calls, or use CUDA graphs. Measure the floor before
// concluding a small shape is inefficient.

__global__ void AbhEmptyKernel() {}

inline void ReportLaunchFloor(dim3 grid, dim3 block, int warmup = 20, int iters = 200) {
    std::printf("== launch floor (empty kernel, grid %ux%ux%u block %ux%u)\n", grid.x, grid.y,
                grid.z, block.x, block.y);
    for (int n : {1, 2, 4, 8}) {
        const Timing t = TimeIt(
            [&](int) {
                for (int i = 0; i < n; ++i) AbhEmptyKernel<<<grid, block>>>();
            },
            warmup, iters);
        std::printf("   %d launches  min=%.4f ms  -> %.4f ms/launch\n", n, t.min_ms, t.min_ms / n);
    }
    std::printf("   (a single launch of grid 1x1 gives the pure API+dispatch cost;\n");
    std::printf("    the difference from the real grid is block dispatch throughput)\n\n");
}

}  // namespace abh

#endif  // ABH_AB_HARNESS_CUH_
