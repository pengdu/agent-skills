# Streams 与并发（copy-compute 重叠、多 stream、CUDA Graphs）

提炼自 NVIDIA CUDA C++ Best Practices Guide 与 Programming Guide 的并发章节。

## 目录
1. 并发的前提条件
2. Copy-Compute 重叠模式
3. 多 stream 分块流水线
4. Stream 语义与常见陷阱
5. 事件与跨 stream 依赖
6. Stream 优先级
7. CUDA Graphs 降低启动开销
8. 多 GPU 并发
9. 诊断：用 nsys 验证重叠
10. 并发失效排查清单

---

## 1. 并发的前提条件

真正的 H2D/D2H 与 kernel 重叠必须同时满足：

| 条件 | 说明 |
|------|------|
| Pinned host memory | `cudaMallocHost` / `cudaHostAlloc`。pageable 内存的 `cudaMemcpyAsync` 会退化为同步（或经过额外 staging buffer） |
| 异步 API | `cudaMemcpyAsync`，且传入非默认 stream |
| 非默认 stream | legacy default stream（stream 0）与其他 stream 有隐式同步 |
| 设备支持 | `deviceOverlap` / `asyncEngineCount >= 1`；`asyncEngineCount == 2` 才能 H2D 与 D2H 同时进行 |
| 无隐式同步操作 | 见第 4 节陷阱列表 |

Kernel 之间的并发执行还要求：GPU 资源（SM、寄存器、smem）未被单个 kernel 占满。一个占满全部 SM 的 kernel 不会与其他 kernel 并发。

## 2. Copy-Compute 重叠模式

顺序版（无重叠）：

```cpp
cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(d_in, d_out, N);
cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
```

重叠版（分块 + 多 stream）：

```cpp
constexpr int NS = 4;
cudaStream_t streams[NS];
for (int i = 0; i < NS; ++i) cudaStreamCreate(&streams[i]);

int chunk = N / NS;
for (int i = 0; i < NS; ++i) {
    int off = i * chunk;
    cudaMemcpyAsync(d_in + off, h_in + off, chunk * sizeof(float),
                    cudaMemcpyHostToDevice, streams[i]);
    kernel<<<grid_chunk, block, 0, streams[i]>>>(d_in + off, d_out + off, chunk);
    cudaMemcpyAsync(h_out + off, d_out + off, chunk * sizeof(float),
                    cudaMemcpyDeviceToHost, streams[i]);
}
for (int i = 0; i < NS; ++i) cudaStreamSynchronize(streams[i]);
```

要点：
- `h_in` / `h_out` 必须是 pinned memory。
- 分块数经验值 2–8：太少重叠不足，太多则每块太小、启动/拷贝开销占比升高。
- 理想时间 ≈ max(总拷贝时间, 总计算时间) + 一个块的流水线填充；相对顺序版上限约 2–3x（取决于 copy:compute 比例与 copy engine 数量）。
- 发射顺序建议 breadth-first（如上，按块循环依次发 H2D+kernel+D2H），避免老架构上 WDDM/队列造成的假串行。

## 3. 多 stream 分块流水线（进阶）

当单次数据超显存或需持续流式处理时，用固定 stream 池 + 环形缓冲：

```cpp
for (int b = 0; b < numBatches; ++b) {
    int s = b % NS;
    cudaStreamSynchronize(streams[s]);   // 等待该 slot 上一轮完成
    fill_host_buffer(h_buf[s], b);       // CPU 侧准备数据（可与 GPU 重叠）
    cudaMemcpyAsync(d_buf[s], h_buf[s], bytes, cudaMemcpyHostToDevice, streams[s]);
    process<<<g, blk, 0, streams[s]>>>(d_buf[s], d_res[s]);
    cudaMemcpyAsync(h_res[s], d_res[s], rbytes, cudaMemcpyDeviceToHost, streams[s]);
}
```

- 每个 stream 配独立的 host/device 缓冲，避免跨 stream 数据竞争。
- 用 `cudaStreamSynchronize`（或 event）做 per-slot 回收，而非全局 `cudaDeviceSynchronize`。
- CPU 端的填充/后处理天然与 GPU 工作重叠——这常是免费的额外并发。

## 4. Stream 语义与常见陷阱

**隐式同步操作**（会破坏并发，避免出现在热路径中）：
- `cudaMalloc` / `cudaFree`（用内存池 `cudaMallocAsync`/`cudaFreeAsync` 或预分配代替）
- `cudaMemcpy`（同步版）、pageable 内存上的 `cudaMemcpyAsync`
- `cudaMemset`（用 `cudaMemsetAsync`）
- `cudaDeviceSynchronize`、`cudaStreamSynchronize`（放到必要的边界）
- L1/shared 配置切换、`cudaFuncSetCacheConfig`
- legacy default stream 上的任何操作

**Default stream 语义**：
- Legacy（默认）：stream 0 与所有阻塞 stream 互相隐式同步。
- Per-thread：编译加 `--default-stream per-thread`，每个 host 线程独立默认 stream。
- 显式建非阻塞 stream：`cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)`，不与 legacy default stream 同步——多 stream 并发场景推荐。

**WDDM（Windows）**：命令会被批量提交，可用 `cudaStreamQuery(s)` 强制 flush；Linux/TCC 无此问题。

## 5. 事件与跨 stream 依赖

用 event 表达 stream 间依赖，避免全局同步：

```cpp
cudaEvent_t done;
cudaEventCreateWithFlags(&done, cudaEventDisableTiming);  // 不计时更轻量

producer<<<g, b, 0, s1>>>(...);
cudaEventRecord(done, s1);
cudaStreamWaitEvent(s2, done, 0);   // s2 等待 s1 到达该点，GPU 侧等待，不阻塞 CPU
consumer<<<g, b, 0, s2>>>(...);
```

- 计时用 event：`cudaEventCreate`（保留 timing）+ `cudaEventElapsedTime`。
- `cudaEventDisableTiming` 用于纯依赖，开销更低。
- `cudaLaunchHostFunc` 可在 stream 完成点回调 host 函数（回调中禁止调 CUDA API）。

## 6. Stream 优先级

延迟敏感的小任务与吞吐型大任务共存时：

```cpp
int lo, hi;
cudaDeviceGetStreamPriorityRange(&lo, &hi);   // hi 数值更小 = 优先级更高
cudaStreamCreateWithPriority(&s_latency, cudaStreamNonBlocking, hi);
cudaStreamCreateWithPriority(&s_bulk,    cudaStreamNonBlocking, lo);
```

优先级影响 block 调度顺序，但不抢占已在运行的 block；对长 kernel 的干扰有限。

## 7. CUDA Graphs 降低启动开销

大量短 kernel（<10–20 μs）时，launch 开销与 CPU 提交成为瓶颈。用 stream capture 把现有多 stream 流水线固化为 graph：

```cpp
cudaGraph_t graph;
cudaGraphExec_t exec;
cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);
// ... 原有的 memcpyAsync / kernel / event 依赖，照常发射（可跨多 stream，经 event 汇入被 capture 的 stream）
cudaStreamEndCapture(s1, &graph);
cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

for (int iter = 0; iter < steps; ++iter)
    cudaGraphLaunch(exec, s1);        // 单次提交整个 DAG
```

- 适用：每次迭代拓扑固定的循环（训练 step、推理 pipeline）。
- 参数变化用 `cudaGraphExecKernelNodeSetParams` 更新，避免重新 instantiate。
- Graph 内部依赖是 DAG，运行时自动并发无依赖节点——比手工 stream 编排更容易吃满并发。

## 8. 多 GPU 并发

```cpp
for (int dev = 0; dev < nDev; ++dev) {
    cudaSetDevice(dev);
    cudaMemcpyAsync(..., streams[dev]);
    kernel<<<g, b, 0, streams[dev]>>>(...);
}
```

- `cudaSetDevice` + 每设备独立 stream，天然并发。
- P2P：`cudaDeviceCanAccessPeer` → `cudaDeviceEnablePeerAccess` → `cudaMemcpyPeerAsync`（NVLink/PCIe 直传，绕过 host）。
- 一个 pinned buffer 对所有设备可用（`cudaHostAlloc` + `cudaHostAllocPortable`）。

## 9. 诊断：用 nsys 验证重叠

```bash
nsys profile -o report --stats=true ./app
nsys stats report.nsys-rep   # 查看 cuda_gpu_kern_sum / cuda_gpu_mem_time_sum
```

在 GUI（nsys-ui）时间线上检查：
- 各 stream 行上 kernel 与 memcpy 是否在时间上重叠。
- Copy engine（HtoD/DtoH 行）是否与 compute 行并行。
- 大段空白 = CPU 提交瓶颈或隐式同步。

量化重叠率：`重叠率 = 1 - 实际总时长 / (拷贝时长 + 计算时长)`，接近 0 说明完全串行。

## 10. 并发失效排查清单

按顺序检查：

1. host 内存是 pinned 吗？（最常见原因）
2. 用的是 `cudaMemcpyAsync` 且传了非默认 stream？
3. 有没有混入 legacy default stream 的操作（stream 0 上的 launch、同步版 API）？
4. 循环里有 `cudaMalloc`/`cudaFree`/`cudaDeviceSynchronize` 吗？
5. stream 是 `cudaStreamNonBlocking` 创建的吗？
6. kernel 是否占满整卡（grid 太大导致 kernel 间无并发空间）？
7. `asyncEngineCount` 是多少？只有 1 时 H2D 与 D2H 无法互相重叠。
8. Windows WDDM？尝试 `cudaStreamQuery` flush 或换 TCC 模式。
9. 每块数据量是否太小（<1 MB 拷贝、<10 μs kernel），开销掩盖了重叠收益？→ 考虑 CUDA Graphs 或增大分块。
