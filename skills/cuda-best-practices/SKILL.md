---
name: cuda-best-practices
description: CUDA C++ 开发最佳实践经验库，提炼自 NVIDIA CUDA C++ Best Practices Guide / Programming Guide。当编写、审查或优化 CUDA 程序时使用，覆盖三大领域：(1) Stream 并发——多 stream copy-compute 重叠、pinned memory、事件依赖、CUDA Graphs、多 GPU；(2) 内存优化——host-device 传输、合并访问、shared memory tiling、bank conflict、Unified Memory、L2 persistence；(3) Kernel 调优——occupancy、延迟隐藏、ILP、分支发散、warp 原语、原子操作。触发场景：CUDA 代码性能问题、"kernel 太慢"、"拷贝和计算怎么重叠"、"多 stream 不并发"、"occupancy 低"、"访存优化"、CUDA 代码 review 等。
---

# CUDA 开发最佳实践

## Overview

NVIDIA 官方指南中经验规则的结构化提炼，用于编写高性能 CUDA 代码和诊断性能问题。按问题域组织为三个 reference 文档，按需加载对应文档。

## 与 cuda-auto-tune 的分工

- 本 skill：**经验规则与代码模式**——写代码时的前置指导、review 时的判断依据、无 profiler 环境下的诊断。
- `cuda-auto-tune` skill：**NCU 驱动的迭代调优闭环**——已有可运行 kernel、能采集 Nsight Compute 数据时优先用它。
- 两者结合：用 cuda-auto-tune 定位瓶颈指标，用本 skill 的对应文档找修复模式。

## 问题路由

根据用户问题选择加载的文档：

| 问题特征 | 加载文档 |
|---------|---------|
| 多 stream、异步拷贝、copy-compute 重叠、并发失效、pinned memory、CUDA Graphs、多 GPU、launch 开销大 | [references/streams-concurrency.md](references/streams-concurrency.md) |
| 传输慢、带宽利用低、合并访问、AoS/SoA、shared memory、bank conflict、Unified Memory、L2 缓存、寄存器溢出、向量化访存 | [references/memory-optimization.md](references/memory-optimization.md) |
| kernel 慢、block/grid 怎么选、occupancy、warp stall、分支发散、归约、原子操作、指令优化、循环展开 | [references/kernel-tuning.md](references/kernel-tuning.md) |

跨域问题（如"整个程序优化"）按优先级顺序处理：**减少/重叠数据传输 → 访存优化 → kernel 内部优化**。

## 核心原则（任何场景都适用）

1. **先测量再优化**：定性瓶颈类型（memory/compute/latency-bound）后再动手；memory-bound 的 kernel 做指令优化无效。
2. **每次只改一处并复测**：避免多个改动互相掩盖效果。
3. **优先级**：host-device 传输（PCIe 是最窄的管道）> 全局内存访问模式 > 片上内存利用 > 指令级优化。
4. **异步是默认姿态**：pinned memory + `cudaMemcpyAsync` + 非默认 stream；热路径禁止 `cudaMalloc`/`cudaFree`/`cudaDeviceSynchronize`。
5. **高 occupancy 不是目标**，隐藏延迟才是；50% occupancy + 高 ILP 常胜过 100% occupancy + 寄存器紧张。

## 快速诊断入口

用户报"多 stream 没有并发"时，直接按 streams-concurrency.md 第 10 节排查清单逐条检查（pinned memory → async API → default stream 污染 → 隐式同步 → 资源占满）。

用户报"kernel 慢"但无 profile 数据时，先建议采集：

```bash
nsys profile -o report --stats=true ./app        # 系统级：stream 重叠、传输占比
ncu --set full -o kernel_report ./app            # kernel 级：交给 cuda-auto-tune 分析
```
