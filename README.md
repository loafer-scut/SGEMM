# CUDA SGEMM Optimization Journey

基于 CC 12.0 架构的 CUDA 单精度矩阵乘法（SGEMM）逐步优化。涵盖了不同数据规模的全面性能压测，以及底层硬件瓶颈的深度剖析。

## 硬件环境
- **GPU**: NVIDIA CC 12.0 (WSL2)
- **L2 Cache**: 约 72MB
- **测试脚本**: [`benchmark.sh`](./benchmark.sh) (基于 20 个数据点的自动化对比测试，用于对比基准线和第一次tiling优化)

---

## 优化版本演进 & 性能数据

### v1 Naive SGEMM (基准线)
- **代码**: [`naive.cu`](./naive.cu)
- **耗时**: 约**18.73 ms** (以 N=1024 为例，约 57 GFLOPS)
- **瓶颈诊断** (ncu 采样):
  - `Compute (SM) Throughput`: 约**95%** (超载)
  - `L2 Hit Rate`: 约**94.94%** (极高)
  - `Memory Throughput (DRAM)`: 约**0.63%** (极低)
- **核心结论**:
  > **瓶颈不在 DRAM 带宽，也不在 L2 缓存，而在 LSU（加载/存储单元）。**
  > 因为每个线程算 1 个元素，每次 FMA 却要发 2 条全局加载指令。SM 内部的 LSU 流水线被撑爆，FMA 计算单元在闲置。
  > **注意**：当前 ncu 报告的 `Compute (SM) Throughput` 达到 95%，并不是 FMA 真的忙，而是 LSU 的占有率极高。这提醒我们不要被单一指标欺骗。

### v2 Shared Memory Tiling (分块优化)
- **代码**: [`shared_memory.cu`](./shared_memory.cu)
- **优化策略**: 引入 Shared Memory 缓存 A 和 B 的 Tile（32x32），将 `O(N^3)` 的全局访存降为 `O(N^2)`，极大减轻了 LSU 压力和 DRAM 带宽压力，实现内存数据的复用。
- **核心结论**:
  > 在小尺寸（N ≤ 1536）时，数据完美驻留 L2，但Tiling 引入的 `__syncthreads()` 同步开销导致其略慢于 Naive，即shared_memory提供的加速小于同步开销。
  > 当尺寸突破 L2 缓存容量（N ≥ 3072）时，Tiling 实现全面反超。在 N=8192 时，Naive 耗时 **1216ms**，而 Tiling 仅需 **872ms**，性能提升约 40%。

---

## 深度性能压测：可扩展性分析 (Scalability Analysis)

为了探究不同问题规模下的硬件表现，我们使用 [`naive_shared_plot.py`](./naive_shared_plot.py) 绘制了 20 个数据点的双对数折线图：

![Performance Scalability](docs/scalability_20_points.png)

**图表三大阶段解析**：
1. **缓存红利期 (N=512 ~ 1024)**：矩阵极小（<12MB），全量驻留 L2，Naive 免去同步开销，短暂领先。
2. **临界震荡期 (N=1536 ~ 2560)**：数据量逼近 L2 极限，Naive 版本出现强烈的 **Cache Thrashing (缓存颠簸)**，耗时呈锯齿状波动（图中红线折角极其明显），而 Tiling 版本依靠片上复用，平稳穿越风暴区。
3. **DRAM 灾难期 (N > 4096)**：彻底击穿 L2 缓存，Naive 的 O(N^3) 重复访存将 DRAM 带宽打满。Tiling 展现出强大的降维打击能力，两者差距呈对数级拉开。

---

## ncu 硬件指标验证 (Physical Profiling)

为了印证图表中的拐点，我们在两个关键尺寸下抓取了底层硬件计数器（详见仓库内的 `ncu_*.txt`）：

| 版本 / 尺寸 | L2 Hit Rate | DRAM Throughput | Kernel Time (ncu) |
| :--- | :--- | :--- | :--- |
| Naive (N=2048) | **97.60%** | **0.96%** | 1041 ms |
| Tiling (N=2048) | **97.60%** | **1.24%** | 889 ms |
| Naive (N=8192) | **49.67%** | **14.94%** | 4636 ms |
| Tiling (N=8192) | **49.68%** | **20.87%** | 3480 ms |

**硬核数据剖析**：
*   **在 N=2048 时**：48MB 矩阵完美填入 L2，L2 命中率高达 97.6%，DRAM 几乎在睡觉。此时 Tiling 略慢是因为同步指令的微小开销。
*   **在 N=8192 时**：矩阵达 768MB，L2 命中率腰斩至 49%。**注意 Tiling 的 DRAM Throughput 高达 20.87%**（远超 Naive 的 14.94%）。这证明 Tiling 的合并访存能高效榨干 DRAM 带宽，避免总线陷入无效的细碎请求中，从而节省了超过 1 秒的计算时间。

---

## 下一步计划 (Roadmap)
- [ ] **解决 Bank Conflict**：目前在读取 `As` 和 `Bs` 时存在 32 路冲突。计划引入 `[32][33]` 的 Padding 技巧，进一步压榨 SM 内部吞吐。
- [ ] **Register Tiling**：提高单线程的计算/访存比（ILP），向 1 TFLOPS 发起冲击。
- [ ] **128-bit 向量化**：引入 `float4` 访存，减少指令发射开销。
