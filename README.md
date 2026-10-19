# CUDA SGEMM 在 RTX 5060 Laptop 上的调优

本仓库记录 SGEMM 从 naive 实现到 4×4 寄存器分块的完整优化过程，所有性能数字都可以用
`benchmark.sh` 复现，原始数据在 `results/`。最终在 RTX 5060 Laptop（sm_120，L2 = 32 MB）
上达到 5931 GFLOPS，约为 naive 版本的 5.3 倍。

![Scalability](docs/scalability_20_points.png)

## 各版本的峰值吞吐

| 版本 | 峰值 GFLOPS | 出现位置 | 相对 naive |
| :--- | ---: | ---: | ---: |
| naive | 1125 | N=2560 | 1.00× |
| shared memory 分块 | 1454 | N=2048 | 1.29× |
| 4×4 寄存器分块 | 5931 | N=2560 | 5.27× |

## 每次改动值多少（统一在 N=1024 对比）

| 改动 | GFLOPS | 相对上一版 |
| :--- | ---: | ---: |
| naive，一个线程算一个数 | 1104 | — |
| 换成 shared memory 分块 | 1435 | 1.30× |
| 换成 4×4 寄存器分块 | 5591 | 3.90× |
| 其中：给 shared memory 加 padding（`PAD=1`） | 5596 | 1.001× |

最后一行是个有用的负结果：理论上有 4 路 bank conflict，但这里实测只有 0.08% 的差别，
说明此版本并不受 shared memory 带宽限制。padding 仍然保留，因为成本为零。

## 文件

| 文件 | 说明 |
| :--- | :--- |
| `naive.cu` | 最直接的实现，一个线程算一个输出 |
| `shared_memory.cu` | 用 shared memory 做 32×32 分块 |
| `register_tiling.cu` | 4×4 寄存器分块（当前最快） |
| `devinfo.cu` | 打印 GPU 型号、L2、SM 数等硬件参数 |

## 怎么跑

```bash
# 单点：编译并运行 N=1024
nvcc -O3 -arch=sm_120 -DDSIZE=1024 register_tiling.cu -o build/v3_1024 && ./build/v3_1024

# 正确性自检（N ≤ 256 时启用 CPU 参考对照）
nvcc -O3 -arch=sm_120 -DDSIZE=128 register_tiling.cu -o build/v3_128 && ./build/v3_128

# 全套 20 个尺寸 → 写入 results/*.csv，再生成曲线
bash benchmark.sh && python3 plot_scalability.py
```

## 三个版本在做什么

- **naive**：一个线程算一个数，每做一次乘加都要从全局内存读两次数据，慢在访存。
- **shared memory 分块**：先把一块数据搬进 shared memory，读取的位置变近了，但每次乘加
  仍然要读两次，所以只快了 29%。
- **4×4 寄存器分块**：一个线程一次算 16 个数，读 8 次就能算 16 次乘加——每单位计算需要的
  读取次数降到原来的 1/4，而且 16 条指令互不依赖，延迟隐藏做的更好，所以快 3.9 倍。

数据访问的层次是寄存器 > shared memory > 全局内存，速度依次下降。优化的核心就是让数据
尽量待在更高层、并且被复用更多次，而不是每次都回去找。

## 关键现象：32 MB 是分界线

数据量小的时候（单个矩阵小于 32 MB），三个版本都很稳定；一旦超过 32 MB，曲线同时掉台阶。

这个 32 MB 可以用数据卡出来：

| N | 单矩阵大小 | naive | 4×4 寄存器分块 |
| ---: | ---: | ---: | ---: |
| 2560 | 26.2 MB | 1125 | 5931 |
| 3072 | 37.7 MB | 837 | 5515 |

26.2 MB 还很平稳、37.7 MB 立刻掉，说明分界线在两者之间；再用 `devinfo.cu` 打印
`l2CacheSize`，得到 32.0 MB，正好落在这个区间里。两边对上了，解释成立

## 测量方法

一开始是单次计时，忽略了首次启动的 JIT 编译开销，把 1.95 ms 的 kernel 测成了 5.43 ms，
据此得出的"小尺寸下分块更慢"等结论都是错的。现在的方式是：**预热 3 次，再重复 20 次
取最小值**（同时记录平均值，两者差得多说明有降频）。

## 环境

RTX 5060 Laptop（8 GB，sm_120），L2 = 32 MB，CUDA 13.x，WSL2，
编译参数 `nvcc -O3 -arch=sm_120`。

## License

MIT
