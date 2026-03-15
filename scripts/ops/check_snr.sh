#!/bin/bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv-linux/bin/activate
python3 << 'EOF'
import pyarrow.parquet as pq
import numpy as np

cat = pq.read_table("04_GOLD/sovereign_catalog.parquet")
snr = cat.column("snr_score").to_pylist()
valid = [s for s in snr if s is not None and s > 0]
kinds = cat.column("kind").to_pylist()
sizes = cat.column("size_bytes").to_pylist()

print(f"=== SOVEREIGN CATALOG SNR DISTRIBUTION ===")
print(f"Total artifacts:  {len(snr)}")
print(f"With SNR score:   {len(valid)}")
print(f"Without SNR:      {len(snr) - len(valid)}")
print(f"Mean SNR:         {np.mean(valid):.4f}")
print(f"Median SNR:       {np.median(valid):.4f}")
print(f"P90 SNR:          {np.percentile(valid, 90):.4f}")
print(f"P95 SNR:          {np.percentile(valid, 95):.4f}")
print(f"Above 0.85:       {sum(1 for s in valid if s >= 0.85)} ({sum(1 for s in valid if s >= 0.85)/len(valid)*100:.1f}%)")
print(f"Above 0.90:       {sum(1 for s in valid if s >= 0.90)} ({sum(1 for s in valid if s >= 0.90)/len(valid)*100:.1f}%)")
print(f"Above 0.95:       {sum(1 for s in valid if s >= 0.95)} ({sum(1 for s in valid if s >= 0.95)/len(valid)*100:.1f}%)")
print()
print(f"=== BY ARTIFACT KIND ===")
from collections import Counter, defaultdict
kind_snr = defaultdict(list)
for k, s in zip(kinds, snr):
    if s is not None and s > 0:
        kind_snr[k].append(s)
for k, scores in sorted(kind_snr.items(), key=lambda x: -np.mean(x[1]))[:15]:
    print(f"  {k:20s}: n={len(scores):>6} mean={np.mean(scores):.3f} p50={np.median(scores):.3f}")
print()
# Total size
print(f"=== DATA VOLUME ===")
total_gb = sum(s for s in sizes if s) / 1e9
print(f"Total data: {total_gb:.2f} GB across {len(snr)} artifacts")
EOF
