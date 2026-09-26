# 内存优化（数据传输、全局/共享/常量内存、Unified Memory）

提炼自 NVIDIA CUDA C++ Best Practices Guide 内存章节。优先级排序：**减少 host-device 传输 > 合并访问 > 利用片上内存 > 其他**。

## 目录
1. Host-Device 传输
2. Pinned / Mapped / Unified Memory 选型
3. 全局内存合并访问
4. Shared memory：tiling 与 bank conflict
5. L2 persistence（Ampere+）
6. 常量内存、只读缓存、寄存器
7. 向量化访存
8. 分配策略与内存池
9. 精度与带宽

---

## 1. Host-Device 传输

PCIe/NVLink 带宽远低于显存带宽（PCIe 4.0 x16 ≈ 32 GB/s vs HBM 数百 GB/s~TB/s），传输是第一优化对象。

- **能不传就不传**：中间结果留在 device；即使某步在 GPU 上不划算，若能避免一来一回的传输，整体也可能更快。
- **合并小传输**：多次小 memcpy 合成一次大块传输（打包进连续 buffer），每次 memcpy 有 ~几 μs 固定开销。
- **重叠传输与计算**：见 [streams-concurrency.md](streams-concurrency.md)。
- 衡量标准：程序有效带宽 vs 理论带宽；`nsys` 的 `cuda_gpu_mem_time_sum` 看传输占比。

## 2. Pinned / Mapped / Unified Memory 选型

| 类型 | API | 适用 | 注意 |
|------|-----|------|------|
| Pageable | `malloc` | 不参与传输的 host 数据 | Async 拷贝会退化 |
| Pinned | `cudaMallocHost` / `cudaHostAlloc` | 反复传输的 buffer；async 重叠必需 | 过量分配会拖慢 OS，分配本身很贵——复用而非频繁分配释放 |
| Mapped (zero-copy) | `cudaHostAlloc(..., cudaHostAllocMapped)` | 只读一次、访问稀疏的数据；集成显卡 | 每次访问都走 PCIe，重复访问性能差 |
| Unified Memory | `cudaMallocManaged` | 快速原型、超显存数据、访问模式复杂 | 用 `cudaMemPrefetchAsync` + `cudaMemAdvise` 避免 page fault 风暴 |

Unified Memory 调优：

```cpp
cudaMemAdvise(p, bytes, cudaMemAdviseSetPreferredLocation, dev);
cudaMemAdvise(p, bytes, cudaMemAdviseSetReadMostly, dev);      // 多设备只读复制
cudaMemPrefetchAsync(p, bytes, dev, stream);                    // kernel 前预取，消除按页迁移
```

性能敏感路径首选显式 `cudaMalloc` + `cudaMemcpyAsync`（pinned），行为最可控。

## 3. 全局内存合并访问

Warp 内 32 线程的访问会被合并成最少的 32-byte sector 事务。目标：**warp 访问连续对齐的地址**。

```cpp
// 好：相邻线程访问相邻元素（stride-1）
data[blockIdx.x * blockDim.x + threadIdx.x]

// 坏：跨步访问，stride 为 t 时有效带宽约降为 1/t
data[threadIdx.x * stride]
```

- **AoS → SoA**：`struct {float x,y,z;} p[N]` 按字段访问时改为 `float x[N], y[N], z[N]`。
- **2D 数组用 `cudaMallocPitch`/`cudaMemcpy2D`** 保证每行对齐。
- 未对齐偏移（如 `data[idx + 1]`）会多取 sector，小幅损失；跨步访问才是大杀器。
- 诊断：NCU `l1tex__t_sectors.../l1tex__t_requests...` 比率，理想 FP32 stride-1 时为 4；越大越散。
- 无法合并的散乱写 → 先写 shared memory 暂存，再按合并模式写回。

## 4. Shared memory：tiling 与 bank conflict

带宽约为全局内存的 10x+，延迟约 1/20~1/30。用途：块内数据复用、协作加载、重排访问模式。

**Tiling 模板（矩阵乘）**：

```cpp
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];
for (int t = 0; t < K / TILE; ++t) {
    As[ty][tx] = A[row * K + t * TILE + tx];      // 合并读入
    Bs[ty][tx] = B[(t * TILE + ty) * N + col];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
}
```

**Bank conflict**：32 个 bank，同 warp 多线程访问同 bank 的不同地址会串行化。

```cpp
__shared__ float tile[32][32];    // 列访问 tile[threadIdx.x][k] → 32-way conflict
__shared__ float tile[32][33];    // padding +1 消除
```

- 广播（同地址）不算 conflict。
- 诊断：NCU `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`。
- 动态 smem 超 48 KB 需 `cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`。
- 注意 smem 用量是 occupancy 限制因子之一，别为复用过度堆料。

## 5. L2 persistence（Ampere+）

反复访问的小热点数据（如 lookup table）可钉在 L2：

```cpp
cudaStreamAttrValue attr{};
attr.accessPolicyWindow.base_ptr  = d_lut;
attr.accessPolicyWindow.num_bytes = lutBytes;                  // <= persistingL2CacheMaxSize
attr.accessPolicyWindow.hitRatio  = 1.0f;
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, lutBytes);
```

window 大于预留空间时调低 `hitRatio` 避免 thrashing。

## 6. 常量内存、只读缓存、寄存器

- **常量内存**（64 KB，`__constant__`）：warp 内所有线程读同一地址时最优（广播）；发散地址会串行化。
- **只读数据** 用 `const __restrict__` 指针提示编译器走只读缓存（`__ldg` 路径）。
- **寄存器**是最快存储；数组下标若编译期可定，编译器才能放寄存器。寄存器溢出到 local memory（实际在全局内存）严重伤性能——NCU 看 `l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum`，非零即有 spill。缓解：减少每线程状态、`__launch_bounds__`、`-maxrregcount`。

## 7. 向量化访存

一次 128-bit 读写减少指令数并提高带宽利用：

```cpp
float4 v = reinterpret_cast<const float4*>(in)[i];   // 要求 16B 对齐，N 是 4 的倍数或处理尾部
out4[i] = make_float4(f(v.x), f(v.y), f(v.z), f(v.w));
```

`cudaMalloc` 返回的地址天然 256B 对齐；子偏移切片时注意对齐被破坏。

## 8. 分配策略与内存池

- `cudaMalloc`/`cudaFree` 是隐式同步点且慢——**热路径禁止**。
- 循环内分配 → 改用 `cudaMallocAsync`/`cudaFreeAsync`（stream-ordered，自带内存池）：

```cpp
cudaMemPool_t pool;
cudaDeviceGetDefaultMemPool(&pool, dev);
uint64_t threshold = UINT64_MAX;                       // 释放不归还 OS，留在池中
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
cudaMallocAsync(&p, bytes, stream);
cudaFreeAsync(p, stream);
```

- 或者一次性预分配大块自己切分（arena）。

## 9. 精度与带宽

memory-bound kernel 的带宽需求直接由字节数决定：

- 中间量用 `half`/`__nv_bfloat16` 存储、FP32 计算（load 时转换），带宽减半。
- 累加器保持 FP32 以保数值稳定。
- `-use_fast_math` / `__expf` 等 intrinsic 换精度取速度，需验证误差可接受。
