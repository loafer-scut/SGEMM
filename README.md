# CUDA SGEMM Optimization Journey

基于 CC 12.0 架构的 CUDA 单精度矩阵乘法（SGEMM）逐步优化。

## 硬件环境
- GPU: NVIDIA CC 12.0 (WSL2)
- 矩阵大小: 1024 x 1024

## 优化版本演进 & 性能数据

### v1 Naive SGEMM (基准线)
- **代码**: `naive.cu`
- **耗时**: **18.73 ms** (约 57 GFLOPS)
- **ncu 分析报告**: `ncu_naive_report.txt`
- **瓶颈诊断**:
  - `Compute (SM) Throughput`: **95%** (超载)
  - `L2 Hit Rate`: **94.94%** (极高)
  - `Memory Throughput (DRAM)`: **0.63%** (极低)
  - **结论**: 瓶颈不在 DRAM 带宽，也不在 L2 缓存，而在 **LSU（加载/存储单元）**。因为每个线程算 1 个元素，每次 FMA 却要发 2 条全局加载指令。SM 内部的 LSU 流水线被撑爆，FMA 计算单元在闲置，导致
  - 当前的compute throghput 不是我们FMA的运行程度，而是LSU的占有率
- **下一步优化动机**: 引入 Shared Memory Tiling，将 O(N^3) 的全局访存降为 O(N^2)，减轻 LSU 压力，真正体现compute throghput，从而计算算术强度寻找瓶颈。

*(待更新: v2 Shared Memory Tiling)*
