# CUDA SGEMM 在 RTX 5060 Laptop 上的调优

本仓库记录 SGEMM 从 naive 实现到 4×4 寄存器分块的优化过程。所有性能数字都可以用
`benchmark.sh` 复现，原始数据在 `results/`；ncu 数据用「运行与复现」里的命令重新采集，
结果在 `results/ncu/`。最终在 RTX 5060 Laptop（sm_120，L2 = 32 MB）上达到 5931 GFLOPS，
约为 naive 版本的 5.3 倍。

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

最后一行是个有用的负结果：按映射关系推算，`As[ty*4+i][k]` 在 warp 内跨 4 行、行跨距正好
32 个 float，理论上构成 4 路 bank conflict；但实测只差 0.08%。冲突确实存在，只是这一项
不是当前的限速环节（可能因为 `a[i]` 这几次读只占 L1 事务的一小部分，或编译器已做了合并，
待查）。padding 仍保留，成本为零，且在更大的分块下会成为必要条件。

## 文件

| 文件 | 说明 |
| :--- | :--- |
| `naive.cu` | 最直接的实现，一个线程算一个输出 |
| `shared_memory.cu` | 用 shared memory 做 32×32 分块 |
| `register_tiling.cu` | 4×4 寄存器分块（当前最快） |
| `benchmark.sh` | 20 个尺寸自动跑批，写入 `results/*.csv` |
| `plot_scalability.py` | 由 `results/` 生成曲线 |
| `devinfo.cu` | 打印 GPU 型号、L2、SM 数等硬件参数 |

## 运行与复现

```bash
# 0. 设备信息（确认 GPU 型号 / L2 大小 / SM 数）
nvcc -O3 -arch=sm_120 devinfo.cu -o build/devinfo && ./build/devinfo

# 1. 正确性自检（N=128，输出 rel_err 与 PASS/FAIL）
nvcc -O3 -arch=sm_120 -DDSIZE=128 register_tiling.cu -o build/v3_128 && ./build/v3_128

# 2. 编译三个版本（N=1024，供单点测试和 ncu 使用）
nvcc -O3 -arch=sm_120 -DDSIZE=1024 naive.cu          -o build/v1_1024
nvcc -O3 -arch=sm_120 -DDSIZE=1024 shared_memory.cu  -o build/v2_1024
nvcc -O3 -arch=sm_120 -DDSIZE=1024 register_tiling.cu -o build/v3_1024
./build/v3_1024

# 3. 全套 20 个尺寸 → 写入 results/*.csv
bash benchmark.sh

# 4. 由数据生成曲线
python3 plot_scalability.py
```

ncu 数据用下面这条命令采集：

```
ncu --clock-control base --launch-skip 3 --launch-count 1 --section SpeedOfLight --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.pct_of_peak_sustained_active ./build/v1_1024
```

再把最后的 `./build/v1_1024` 换成 `./build/v2_1024`、`./build/v3_1024` 各跑一次。

`--launch-skip 3` 用来跳过前三次预热；`--clock-control base` 把 SM 锁在 1.21 GHz，
所以 ncu 里的 Duration 比 benchmark 慢一倍，**绝对时间不可与 benchmark 混用，只看百分比**。

## 三个版本在做什么

- **naive**：一个线程算一个输出，每做一次乘加都要从全局内存读两次数据。它的 L1/TEX 利用率
  在 N=1024 时就已经到 **97.52%**，是三个版本里唯一真正贴着上限的——问题从来不是"数据太大
  装不下"，而是每次乘加都要发两条加载指令，存取指令与计算指令的比例太大。

- **shared memory 分块**：利用 shared memory 比全局访存快，把 32×32 的 tile 先搬进去，
  全局字节数降到 1/32。但在 N=1024 时 L2 利用率只有 6%，省下来的字节换不到时间；它实际
  赚到的是"同样两次访问，shared 比 global 便宜"（不用走 tag 查找与 miss 处理），L1 从
  97.5% 降到 87.5%，所以小 N 只快 **1.30×**。越过 L2 之后，字节数的优势才兑现，加速比
  升到 **1.61×**（N=8192）/ **1.70×**（N=10240）。
  关键在于：它**没有改变每 FMA 的加载次数**，仍然是 2 次——这也为 v3 提供了方向。

- **4×4 寄存器分块**：一个线程一次算 16 个输出，8 次读取服务 16 次乘加，每 FMA 的加载次数
  从 2 降到 **0.5**，L1 事务量降到 1/4。N=1024 时 1435 → 5591 GFLOPS，**3.90×**，与
  "L1 事务降到 1/4"的预期吻合。同时 16 个累加器让同一线程内的指令不再互相等待
  （`issue_active` 从 21.56% 升到 50.99%），这是能把这部分收益吃满的前提，但不是提速的
  主要原因——主要原因仍然是降低了每 FMA 的加载次数。

数据访问的层次是寄存器 > shared memory > 全局内存，速度依次下降。优化的核心是让数据尽量待在
更高层、并且被复用更多次——但要看清楚当前卡在哪一层，否则可能只降了"单价"却没降"次数"。

## 关键现象：32 MB 是分界线

数据量小的时候（单个矩阵小于 32 MB），三个版本都不受容量限制；一旦超过 32 MB，曲线同时掉台阶。

这个 32 MB 可以用数据卡出来：

| N | 单矩阵大小 | naive | 4×4 寄存器分块 |
| ---: | ---: | ---: | ---: |
| 2560 | 26.2 MB | 1125 | 5931 |
| 3072 | 37.7 MB | 837 | 5515 |

26.2 MB 还很平稳、37.7 MB 立刻掉，说明分界线在两者之间；再用 `devinfo.cu` 打印
`l2CacheSize`，得到 32.0 MB，正好落在这个区间里。两边对上了，这个解释才成立。

顺带一个现象：越过 L2 之后，naive 对 v2 的劣势从 1.30× 拉到 1.70×，因为 naive 的全局流量
是 v2 的 32 倍，超容量后这部分直接变成 DRAM 开支；而 v3 对 v2 的优势几乎不变（3.9× → 4.0×），
因为它降的是 L1 访问次数，与数据装不装得进 L2 无关。

## ncu 数据

把 SM 锁在 1.21 GHz 后采集，N=1024（命令见「运行与复现」，原始输出在 `results/ncu/`）：

| 指标 | v1 | v2 | v3 |
| :--- | ---: | ---: | ---: |
| L1/TEX Cache | 97.52% | 87.45% | 92.55% |
| L2 Cache | 5.94% | 6.14% | 26.11% |
| DRAM | 0.58% | 0.76% | 3.43% |
| Compute (SM) | 95.95% | 86.03% | 71.06% |
| `issue_active` | 37.59% | 21.56% | 50.99% |
| 每 FMA 加载次数（代码统计） | 2 | 2 | 0.5 |

三个版本的 L1 都在 87~97%，而 L2 与 DRAM 一直很低——**瓶颈始终在 L1 这一层**：v2 没有动它，
v3 才动了。

这也解释了为什么 v2 只快 1.30×：它降低的是**每次访问的单价**（shared 比 global 便宜）和
延迟，没有改变"每 FMA 要发两次加载"这件事。

还有一个容易误读的地方：v1 的 Compute 95.95% 与它的 L1 97.52% 几乎相等，v2 的 86.03% 与
87.45% 也几乎相等——看起来像"算力打满了"。实际上 `sm__throughput` 取的是 SM 内**最忙的
那个子单元**的利用率，这里最忙的是加载单元（LSU），不是 FMA。所以"Compute 高"不等于
"在算"，三个版本的实际瓶颈都是 Memory。

## 为什么不先动 ILP

首先看大盘：v1，v2、v3 的 Memory 都超过 80%，说明有资源已经被吃满。这时 ILP、TLP 这类延迟隐藏
的手段可以先放一边——它们只在"既没算满、也没搬满"的时候才是主角。ILP 在这里的作用只是
"确认它不是瓶颈"，确认完就可以放下。

首先说明issue_active 只反映发射槽的使用率，不反映做了多少有用功，v1 的 37.59% 高于 v2 的 21.56%，但 v1 更慢，因为它多出来的指令是全局寻址的算术，也不是当前主要因素——memory。

其次要说清 v2 → v3 里 ILP 为什么变了：4×4 分块让每个线程持有 16 个累加器（`TT×TT = 4×4`
个输出），同一线程内的 16 条累加不再互相等待，v2 中 FFMA 串行等待的开销被消掉了。所以去看
`issue_active` 会发现 v3（50.99%）明显高于 v2（21.56%）——但这是"减少加载次数"这个改动的
副产物，不是我们优化 v2 的出发点。

## 下一步

继续降低每 FMA 的加载次数，把 4×4 扩到 8×8（`TS=64, TT=8`），每次加载可服务 4 次乘加。
但要先有预期：v3 的 Compute 已到 71%，L1 需求减半后浮点单元会先接近上限，所以 8×8 的实际
上限约 **1.4 倍**，而不是 2 倍。这个数字要先用 `--section ComputeWorkloadAnalysis` 确认最忙
的管线是 FMA 还是 LSU 才能定。

另一条不占寄存器的路是 `float4`：它不减加载次数，但能把 L1 事务数降到 1/4（注意此时 `PAD`
必须改为 4 以保证 16 字节对齐）。

## 测量方法

一开始是单次计时，忽略了首次启动的 JIT 编译开销，把 1.95 ms 的 kernel 测成了 5.43 ms，据此
得出的"小尺寸下分块更慢"等结论都是错的。现在的方式是：**预热 3 次，再重复 20 次取最小值**
（同时记录平均值，两者差得多说明有降频）。

## 环境

RTX 5060 Laptop（8 GB，sm_120），L2 = 32 MB，CUDA 13.x，WSL2，
编译参数 `nvcc -O3 -arch=sm_120`。

## License

MIT
