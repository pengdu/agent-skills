# Kernel 性能调优（occupancy、指令优化、同步、控制流）

提炼自 NVIDIA CUDA C++ Best Practices Guide 执行配置与指令优化章节。
如果需要 NCU 驱动的迭代式调优闭环（profile → 分析 → 修改 → 复测），优先使用 `cuda-auto-tune` skill；本文档提供背后的原理与经验规则。

## 目录
1. 优化前先测量（APOD）
2. 执行配置：block/grid 选择
3. Occupancy 与限制因子
4. 延迟隐藏与 ILP
5. 指令级优化
6. 控制流与分支发散
7. 同步开销与 warp 原语
8. 原子操作
9. 循环展开与编译器提示
10. 常见反模式速查

---

## 1. 优化前先测量（APOD）

Assess → Parallelize → Optimize → Deploy 循环。

- 先定性瓶颈：memory-bound / compute-bound / latency-bound（NCU roofline，SM% vs MEM%）。
- memory-bound 的 kernel 做指令优化基本无效，反之亦然。
- 用 Amdahl 定律估上限：优化只影响占总时间 P 的部分，最多加速 1/(1-P)。
- 强烈的经验规则：**每次只改一处，改完复测**。

## 2. 执行配置：block/grid 选择

Block size 经验值：

- 从 **128 或 256** 起步；必须是 warp size（32）的倍数。
- 至少 64；小 block（<64）在多数场景 occupancy 上不去。
- Grid 至少要能铺满所有 SM 的数倍（blocks >= SM 数 × 每 SM 可驻留 block 数），给调度器留余量隐藏尾部效应。
- 元素数远大于线程数时用 grid-stride loop，kernel 对任意 grid 尺寸鲁棒：

```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
     i += blockDim.x * gridDim.x) {
    out[i] = f(in[i]);
}
```

- API 辅助：`cudaOccupancyMaxPotentialBlockSize` 给出启发式最优 block size。

## 3. Occupancy 与限制因子

Occupancy = 活跃 warp 数 / 硬件最大 warp 数。三个限制因子：

| 限制因子 | 表现 | 手段 |
|---------|------|------|
| 寄存器 | regs/thread × threads/block 超 SM 寄存器总量 | `__launch_bounds__(maxThreads, minBlocks)`、`-maxrregcount`、减少每线程状态 |
| Shared memory | smem/block × blocks 超 SM smem | 减小 tile、减少 stage、调整 L1/smem carveout |
| Block/warp 槽位 | block 太小导致 block 数达上限 | 增大 block size |

关键认知：
- **高 occupancy ≠ 高性能**。足够隐藏延迟即可（通常 50%+），继续拉高可能牺牲每线程寄存器反而变慢。
- 低 occupancy + 高 ILP 的 kernel（每线程处理多元素）可以很快。
- `--ptxas-options=-v` 编译时打印 regs/smem 用量；NCU 的 Occupancy section 给出限制因子。

## 4. 延迟隐藏与 ILP

GPU 用海量 warp 切换隐藏延迟（全局内存 ~400-800 cycle）。两条路：

1. **TLP**：更多活跃 warp（提高 occupancy）。
2. **ILP**：单线程内多条独立指令流水。每线程处理多个元素、使用多个独立累加器：

```cpp
float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;   // 4 路独立累加，打破依赖链
for (int i = tid; i < n; i += stride * 4) {
    acc0 += a[i] * b[i];
    acc1 += a[i + stride] * b[i + stride];
    acc2 += a[i + stride*2] * b[i + stride*2];
    acc3 += a[i + stride*3] * b[i + stride*3];
}
```

stall 主因诊断（NCU warp stall sampling）：
- `long_scoreboard` 高 → 等全局内存 → 改善访存/预取/async copy。
- `barrier` 高 → `__syncthreads` 太频繁或 block 太大。
- `wait` 高 → 执行依赖 → 提高 ILP。
- `not_selected` 高 → warp 足够多，说明延迟已隐藏好。

## 5. 指令级优化

- 除法与取模贵：2 的幂用位运算（编译期常量时编译器自动做）。
- `sinf/cosf/expf` 有硬件加速 intrinsic 版：`__sinf/__cosf/__expf`（精度低、速度快）；`-use_fast_math` 全局启用。
- 同时要 sin 和 cos → `sincosf`。
- 幂运算：`x*x` 优于 `powf(x, 2.f)`；`rsqrtf(x)` 优于 `1.f/sqrtf(x)`。
- 避免意外的双精度：字面量写 `3.14f` 而非 `3.14`（FP64 吞吐在消费卡上是 FP32 的 1/32~1/64）。
- 整数索引计算多时考虑用 32-bit（`int`），64-bit 整数运算是多条指令。
- FP16/BF16 GEMM-like 模式 → 上 Tensor Core（WMMA / mma.sync / CUTLASS），见 `cuda-auto-tune`。

## 6. 控制流与分支发散

分支发散：warp 内线程走不同路径时两个路径都执行（predication/串行化）。

- 让分支条件按 warp 对齐：`if (threadIdx.x / 32 < k)` 无发散；`if (threadIdx.x % 2)` 全发散。
- 数据依赖的发散：考虑先排序/分桶使相邻线程走同路径。
- 短小分支编译器会转 predication，无需手工优化。
- 诊断：NCU `smsp__thread_inst_executed_per_inst_executed`（avg active threads / 32）。

## 7. 同步开销与 warp 原语

- `__syncthreads()` 只同步 block 内；能用 warp 级就不要用 block 级。
- Warp 内通信用 shuffle，免 smem 免 barrier：

```cpp
// warp 级归约
for (int off = 16; off > 0; off >>= 1)
    val += __shfl_down_sync(0xffffffff, val, off);
```

- 典型 block 归约：先 warp shuffle 归约，每 warp 结果写 smem，再由第一个 warp 归约——只需 1 次 `__syncthreads`。
- 更复杂的协作用 Cooperative Groups（`cg::tiled_partition<32>`）。
- 注意 Volta+ 独立线程调度：warp 内也要用带 `_sync` 后缀的原语并给对 mask。

## 8. 原子操作

- 高冲突原子（所有线程打同一地址）会串行化 → 先做 block/warp 内归约，再每 block 一次 atomic：

```cpp
// 每个 warp 只发一次 atomicAdd
if ((threadIdx.x & 31) == 0) atomicAdd(&global_sum, warp_sum);
```

- 直方图类：先在 smem 建 per-block 直方图，最后合并到全局。
- FP 原子加不满足结合律，结果可能 run-to-run 不一致——需要确定性时改用归约树。

## 9. 循环展开与编译器提示

- `#pragma unroll N`：展开次数编译期已知的热循环，减少分支开销、暴露 ILP；代价是寄存器压力。
- `__restrict__`：告知指针不别名，解锁加载重排与只读缓存。
- `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)`：让编译器按目标 occupancy 分配寄存器。
- `constexpr`/模板参数替代运行时参数，让分支和循环边界在编译期消解。

## 10. 常见反模式速查

| 反模式 | 后果 | 修复 |
|--------|------|------|
| block size 不是 32 倍数 | 尾部 warp 浪费 | 用 128/256 |
| 每线程大数组（动态下标） | 寄存器放不下 → local memory spill | 改 smem 或拆 kernel |
| 热路径 `cudaMalloc`/`Free` | 隐式同步 + 慢 | 内存池 / 预分配 |
| 双精度字面量 | 意外 FP64 | 加 `f` 后缀 |
| 全 block `__syncthreads` 做 warp 内通信 | 无谓等待 | shuffle |
| 单一全局 atomic 累加 | 串行化 | 分层归约 |
| kernel 太小（<10 μs）且量大 | launch 开销主导 | 合并 kernel / CUDA Graphs（见 [streams-concurrency.md](streams-concurrency.md)） |
| 盲目拉满 occupancy | 寄存器被压缩、spill | 以 NCU stall 数据为准 |
