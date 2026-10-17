#!/usr/bin/env python3
"""生成 SGEMM 可扩展性图表。用法: python3 naive_shared_plot.py"""
import csv, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L2_BYTES = 32 * 1024 * 1024
N_L2 = (L2_BYTES / 4) ** 0.5          # 单矩阵超过 L2 的临界 N ≈ 2896

def load(path):
    n, ms, gf = [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            n.append(int(r["N"])); ms.append(float(r["min_ms"])); gf.append(float(r["gflops"]))
    return n, ms, gf

n1, t1, g1 = load("results/v1_naive.csv")
n2, t2, g2 = load("results/v2_tiling.csv")
p_in  = statistics.mean([g for n, g in zip(n1, g1) if n <= 2560])
p_out = statistics.mean([g for n, g in zip(n1, g1) if n >= 3072])
sp = [a / b for a, b in zip(t1, t2)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(n1, g1, "o-", color="crimson",  label=f"v1 Naive  ({p_in:.0f} → {p_out:.0f} GFLOPS)")
ax1.plot(n2, g2, "s-", color="seagreen", label="v2 Shared Tiling")
ax1.axvline(N_L2, ls="--", c="gray", lw=1)
ax1.text(N_L2 * 1.03, p_out * 1.01, f"L2 = 32 MB\nN ≈ {N_L2:.0f}", fontsize=9, color="gray")
ax1.set_xscale("log")
ax1.set_xlabel("Matrix size N (log)")
ax1.set_ylabel("GFLOPS  (2N³ / t)")
ax1.set_title("Effective throughput vs. problem size")
ax1.grid(True, which="both", alpha=.3); ax1.legend()

ax2.plot(n1, sp, "o-", color="navy")
ax2.axhline(1.0, ls=":", c="gray")
ax2.axvline(N_L2, ls="--", c="gray", lw=1)
ax2.set_xscale("log")
ax2.set_xlabel("Matrix size N (log)")
ax2.set_ylabel("Speedup  (v1 / v2)")
ax2.set_title(f"Tiling speedup: {min(sp):.2f}× – {max(sp):.2f}×")
ax2.grid(True, which="both", alpha=.3)

fig.tight_layout()
fig.savefig("docs/scalability_20_points.png", dpi=150)
print("saved docs/scalability_20_points.png")