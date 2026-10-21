#!/usr/bin/env python3
"""SGEMM 可扩展性图：v1~v5 + cuBLAS 基线"""
import os
import pandas as pd
import matplotlib.pyplot as plt

L2_MB = 32
N_L2 = (L2_MB * 1024 * 1024 / 4) ** 0.5        # 单矩阵超过 L2 的临界 N ≈ 2896

def load(path, tag):
    d = pd.read_csv(path)
    return d.rename(columns={'min_ms': f'ms_{tag}',
                             'gflops': f'gf_{tag}',
                             'avg_ms': f'avg_{tag}'})

df = (load('results/v1_naive.csv', 'v1')
      .merge(load('results/v2_tiling.csv', 'v2'), on='N')
      .merge(load('results/v3_regtile.csv', 'v3'), on='N')
      .merge(load('results/v4_regtile_8x8.csv', 'v4'), on='N')
      .merge(load('results/v5_regtile_128tile.csv', 'v5'), on='N')
      .merge(load('results/cublas_sgemm.csv', 'cb'), on='N'))

for tag in ('v2', 'v3', 'v4', 'v5', 'cb'):
    df[f'sp_{tag}'] = df.ms_v1 / df[f'ms_{tag}']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=200)

series = [
    ('cb', 'cuBLAS FP32 (strict)',      '#000000', 'x', '--'),
    ('v1', 'v1 Naive',                  '#d62728', 'o', '-'),
    ('v2', 'v2 Shared Tiling 32×32',    '#2ca02c', 's', '--'),
    ('v3', 'v3 Register Tiling 4×4',    '#1f77b4', '^', '-'),
    ('v4', 'v4 Register Tiling 8×8',    '#ff7f0e', 'D', '-'),
    ('v5', 'v5 128×128 tile',           '#9467bd', 'v', '-'),
]

# ---- 左：吞吐 ----
for tag, label, color, marker, ls in series:
    peak = df[f'gf_{tag}'].max()
    ax1.plot(df.N, df[f'gf_{tag}'], marker=marker, ls=ls, color=color,
             lw=2 if tag != 'cb' else 2.5, ms=5,
             label=f'{label}  (peak {peak:.0f})')

ax1.axvline(N_L2, color='gray', ls='--', alpha=.7)
ax1.text(N_L2 * 1.03, df.gf_v1.max() * .45,
         f'L2 = {L2_MB} MB\n(N ≈ {N_L2:.0f})', fontsize=10, color='gray')
ax1.set_xscale('log')
ax1.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
ax1.set_ylabel('GFLOPS  (2N³ / t_min)', fontsize=12, fontweight='bold')
ax1.set_title('Effective Throughput', fontsize=13, fontweight='bold')
ax1.grid(True, which='both', ls='--', alpha=.5)
ax1.legend(fontsize=9)


# ---- 右：加速比 ----
for tag, label, color, marker, ls in series:
    if tag == 'v1':
        continue
    ax2.plot(df.N, df[f'sp_{tag}'], marker=marker, ls=ls, color=color, lw=2, ms=5,
             label=f'{tag} / v1  (max {df[f"sp_{tag}"].max():.2f}×)')

ax2.axhline(1.0, color='gray', ls=':')
ax2.axvline(N_L2, color='gray', ls='--', alpha=.7)
ax2.set_xscale('log')
ax2.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup vs. v1 Naive', fontsize=12, fontweight='bold')
ax2.set_title('Speedup over Baseline', fontsize=13, fontweight='bold')
ax2.grid(True, which='both', ls='--', alpha=.5)
ax2.legend(fontsize=9)

plt.tight_layout()
os.makedirs('docs', exist_ok=True)
plt.savefig('docs/scalability_20_points.png')
print('已保存 docs/scalability_20_points.png')