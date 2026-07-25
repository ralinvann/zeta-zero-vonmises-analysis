import numpy as np
import csv
from pathlib import Path

N = 10**6
PROJECT_ROOT = Path(__file__).resolve().parent.parent
zeros_path = PROJECT_ROOT / 'zeros.npy'
TABLES_DIR = PROJECT_ROOT / 'results' / 'tables'
SUMMARY_TABLE = TABLES_DIR / 'spacing_tail_summary.csv'
PERCENTILES_TABLE = TABLES_DIR / 'spacing_tail_percentiles.csv'
THRESHOLDS_TABLE = TABLES_DIR / 'spacing_tail_threshold_counts.csv'
gammas = np.load(zeros_path)[:N]

raw_spacings = np.diff(gammas)

T_mid = (gammas[1:] + gammas[:-1]) / 2
N_prime = (np.log(T_mid / (2*np.pi)) + 1) / (2*np.pi)
unfolded = raw_spacings * N_prime

TABLES_DIR.mkdir(parents=True, exist_ok=True)

with SUMMARY_TABLE.open('w', newline='', encoding='utf-8') as fh:
    writer = csv.writer(fh)
    writer.writerow(["series", "min", "max", "mean", "std", "median"])
    writer.writerow([
        "raw_spacings",
        float(raw_spacings.min()),
        float(raw_spacings.max()),
        float(raw_spacings.mean()),
        float(raw_spacings.std()),
        float(np.median(raw_spacings)),
    ])
    writer.writerow([
        "unfolded_spacings",
        float(unfolded.min()),
        float(unfolded.max()),
        float(unfolded.mean()),
        float(unfolded.std()),
        float(np.median(unfolded)),
    ])

with PERCENTILES_TABLE.open('w', newline='', encoding='utf-8') as fh:
    writer = csv.writer(fh)
    writer.writerow(["percentile", "unfolded_spacing_value"])
    for p in [50, 75, 90, 95, 99, 100]:
        writer.writerow([p, float(np.percentile(unfolded, p))])

with THRESHOLDS_TABLE.open('w', newline='', encoding='utf-8') as fh:
    writer = csv.writer(fh)
    writer.writerow(["threshold_multiple_of_mean", "count"])
    mean_unfolded = unfolded.mean()
    for multiple in [3, 4, 5]:
        writer.writerow([multiple, int(np.sum(unfolded > multiple * mean_unfolded))])

print(f"Saved table: {SUMMARY_TABLE}")
print(f"Saved table: {PERCENTILES_TABLE}")
print(f"Saved table: {THRESHOLDS_TABLE}")