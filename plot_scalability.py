#!/usr/bin/env python3
"""SGEMM 可扩展性图：v1 naive / v2 tiling / v3 register tiling"""
import os
import pandas as pd
import matplotlib.pyplot as plt

L2_MB = 32
N_L2 = (L2_MB * 1024 * 1024 / 4) ** 0.5        # 单矩阵超过 L2 的临界 N ≈ 2896

def load(path, tag):
    d = pd.read_csv(path)
    return d.rename(columns={'min_ms': f'ms_{tag}', 'gflops': f'gf_{tag}'})

df = (load('results/v1_naive.csv',  'v1')
      .merge(load('results/v2_tiling.csv', 'v2'), on='N')
      .merge(load('results/v3_regtile.csv', 'v3'), on='N'))

df['sp_v2'] = df.ms_v1 / df.ms_v2
df['sp_v3'] = df.ms_v1 / df.ms_v3

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

# ---- 左：吞吐 ----
ax1.plot(df.N, df.gf_v1, marker='o', ls='-', color='#d62728', lw=2, ms=5, label='v1 Naive')
ax1.plot(df.N, df.gf_v2, marker='s', ls='-', color='#2ca02c', lw=2, ms=5, label='v2 Shared Tiling')
ax1.plot(df.N, df.gf_v3, marker='^', ls='-', color='#1f77b4', lw=2, ms=5, label='v3 Register Tiling 4x4')
ax1.axvline(N_L2, color='gray', ls='--', alpha=.7)
ax1.text(N_L2 * 1.03, df.gf_v1.max() * .5,
         f'L2 = {L2_MB} MB\n(N ≈ {N_L2:.0f})', fontsize=10, color='gray')
ax1.set_xscale('log')
ax1.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
ax1.set_ylabel('GFLOPS  (2N³ / t_min)', fontsize=12, fontweight='bold')
ax1.set_title('Effective Throughput', fontsize=13, fontweight='bold')
ax1.grid(True, which='both', ls='--', alpha=.5)
ax1.legend(fontsize=10)

# ---- 右：加速比 ----
ax2.plot(df.N, df.sp_v2, marker='s', ls='--', color='#2ca02c', lw=2, ms=5,
         label=f'v2 / v1  (max {df.sp_v2.max():.2f}×)')
ax2.plot(df.N, df.sp_v3, marker='^', ls='-', color='#1f77b4', lw=2, ms=5,
         label=f'v3 / v1  (max {df.sp_v3.max():.2f}×)')
ax2.axhline(1.0, color='gray', ls=':')
ax2.axvline(N_L2, color='gray', ls='--', alpha=.7)
ax2.set_xscale('log')
ax2.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup vs. v1 Naive', fontsize=12, fontweight='bold')
ax2.set_title('Speedup over Baseline', fontsize=13, fontweight='bold')
ax2.grid(True, which='both', ls='--', alpha=.5)
ax2.legend(fontsize=10)

plt.tight_layout()
os.makedirs('docs', exist_ok=True)
plt.savefig('docs/scalability_20_points.png')
print('已保存 docs/scalability_20_points.png')