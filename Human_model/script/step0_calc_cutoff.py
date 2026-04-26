import sys
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

if len(sys.argv) < 2:
    print("Usage: python step2_calc_cutoff.py <overlap_bed_file>")
    sys.exit(1)

bed_file = sys.argv[1]
df = pd.read_table(bed_file, sep='\t', header=None)

# logFC_hdvsgDNA 是第7列 (基于0的索引为6)
logFC = df[6].values

# 1. 还原 R 语言 density() 的核心参数
# R 默认使用 Silverman's rule of thumb 计算带宽 (bw)，并在 [min - 3*bw, max + 3*bw] 范围内生成 512 个等距网格点
kde = gaussian_kde(logFC, bw_method='silverman')
bw = np.sqrt(kde.covariance[0, 0])
x_grid = np.linspace(logFC.min() - 3 * bw, logFC.max() + 3 * bw, 512)

# 2. 完全复现 R 脚本中的 quantile(d$x, probs=0.5)
mean_cutoff2 = np.quantile(x_grid, 0.5)

print(f"{mean_cutoff2:.15f}")