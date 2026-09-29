// Achievable DRAM bandwidth on the current device.
//
// This is the roofline denominator. The spec-sheet figure (2 * memClk * busWidth)
// is not reachable by any real kernel; on a 2080 Ti it reads 616 GB/s while a
// float4 copy measures 527. Using the spec number understates every kernel's
// achieved efficiency and invents headroom that does not exist.
//
//   nvcc -O3 -arch=sm_XX bandwidth_probe.cu -o bw && ./bw
//
// Options:
//   --mb N        buffer size per side in MiB (default 256)
//   --block N     threads per block (default 256)
//   --waves N     blocks per SM (default 32)
//   --iters N     timed iterations (default 200)
//   --warmup N    untimed iterations (default 20)
//
// Report the *copy* number for kernels that both read and write, the *read*
// number for read-dominated kernels, and the *write* number for write-dominated
// ones. Buffers default to 256 MiB, tens of times any current L2, so nothing is
// served from cache.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(call)                                                                    \
    do {                                                                            \
        cudaError_t e_ = (call);                                                     \
        if (e_ != cudaSuccess) {                                                     \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,      \
                         cudaGetErrorString(e_));                                    \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

// Grid-stride so the block count is independent of the buffer size.
__global__ void BwRead(const float4* __restrict__ src, size_t n, float* sink) {
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const float4 v = src[i];
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
    }
    // Data-dependent and never true: the loads cannot be eliminated, but no
    // store traffic is generated either.
    if (acc.x == 1.2345678e30f) sink[0] = acc.x + acc.y + acc.z + acc.w;
}

__global__ void BwWrite(float4* __restrict__ dst, size_t n) {
    const float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        dst[i] = v;
    }
}

__global__ void BwCopy(const float4* __restrict__ src, float4* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

// Scalar variants, to show what a kernel loses by not vectorising its accesses.
__global__ void BwCopyScalar(const float* __restrict__ src, float* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

struct Timing {
    double mean_ms;
    double min_ms;
};

template <typename Fn>
Timing TimeIt(Fn&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<float> s(iters);
    for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(a));
        fn();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&s[i], a, b));
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    double sum = 0;
    for (float v : s) sum += v;
    return Timing{sum / iters, *std::min_element(s.begin(), s.end())};
}

int main(int argc, char** argv) {
    size_t mb = 256;
    int block = 256, waves = 32, iters = 200, warmup = 20;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--mb")) mb = (size_t)std::atol(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--block")) block = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--waves")) waves = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--warmup")) warmup = std::atoi(argv[i + 1]);
    }

    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, 0));
    CK(cudaSetDevice(0));
    const double spec = 2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8) / 1e9;
    std::printf("device : %s  SM%d%d  SMs=%d\n", p.name, p.major, p.minor, p.multiProcessorCount);
    std::printf("L2     : %.2f MiB\n", p.l2CacheSize / 1048576.0);
    std::printf("spec   : %.1f GB/s  (2 x %.0f MHz x %d bit -- NOT achievable)\n", spec,
                p.memoryClockRate / 1000.0, p.memoryBusWidth);

    const size_t bytes = mb << 20;
    if (bytes < (size_t)p.l2CacheSize * 8) {
        std::printf("WARNING: buffer is only %.1fx L2; raise --mb or results include cache hits\n",
                    (double)bytes / p.l2CacheSize);
    }
    const size_t n4 = bytes / sizeof(float4);
    const size_t n1 = bytes / sizeof(float);
    const int grid = p.multiProcessorCount * waves;

    float4 *a = nullptr, *b = nullptr;
    float* sink = nullptr;
    CK(cudaMalloc(&a, bytes));
    CK(cudaMalloc(&b, bytes));
    CK(cudaMalloc(&sink, sizeof(float)));
    CK(cudaMemset(a, 0x3f, bytes));

    std::printf("config : %zu MiB/side, %d blocks x %d thr, warmup=%d iters=%d\n\n", mb, grid,
                block, warmup, iters);
    std::printf("%-14s %10s %10s %12s %12s %8s\n", "pattern", "mean_ms", "min_ms", "GB/s(min)",
                "GB/s(mean)", "of spec");

    struct Row {
        const char* name;
        double moved;
        Timing t;
    };
    std::vector<Row> rows;
    rows.push_back({"read f4", (double)bytes,
                    TimeIt([&] { BwRead<<<grid, block>>>(a, n4, sink); }, warmup, iters)});
    rows.push_back({"write f4", (double)bytes,
                    TimeIt([&] { BwWrite<<<grid, block>>>(b, n4); }, warmup, iters)});
    rows.push_back({"copy f4", 2.0 * bytes,
                    TimeIt([&] { BwCopy<<<grid, block>>>(a, b, n4); }, warmup, iters)});
    rows.push_back({"copy f1", 2.0 * bytes,
                    TimeIt(
                        [&] {
                            BwCopyScalar<<<grid, block>>>((const float*)a, (float*)b, n1);
                        },
                        warmup, iters)});
    CK(cudaGetLastError());

    double copy_f4 = 0;
    for (const Row& r : rows) {
        const double g_min = r.moved / (r.t.min_ms * 1e-3) / 1e9;
        const double g_mean = r.moved / (r.t.mean_ms * 1e-3) / 1e9;
        std::printf("%-14s %10.4f %10.4f %12.1f %12.1f %7.1f%%\n", r.name, r.t.mean_ms, r.t.min_ms,
                    g_min, g_mean, 100.0 * g_min / spec);
        if (!std::strcmp(r.name, "copy f4")) copy_f4 = g_min;
    }

    std::printf("\nUse for roofline denominators:\n");
    std::printf("  read-only kernels   : %.1f GB/s\n", rows[0].moved / (rows[0].t.min_ms * 1e-3) / 1e9);
    std::printf("  write-only kernels  : %.1f GB/s\n", rows[1].moved / (rows[1].t.min_ms * 1e-3) / 1e9);
    std::printf("  read+write kernels  : %.1f GB/s   <- the usual one\n", copy_f4);

    CK(cudaFree(a));
    CK(cudaFree(b));
    CK(cudaFree(sink));
    return 0;
}
