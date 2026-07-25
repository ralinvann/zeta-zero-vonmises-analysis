from pathlib import Path
import struct
from math import log2
import csv

import mpmath
import numpy as np

mpmath.mp.prec = 300

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / 'data'
TABLES_DIR = PROJECT_ROOT / 'results' / 'tables'
ZEROS_OUTPUT = PROJECT_ROOT / 'zeros.npy'

files = [
    'zeros_14.dat',
    'zeros_5000.dat',
    'zeros_26000.dat',
    'zeros_236000.dat',
    'zeros_446000.dat',
]

EPS = mpmath.mpf(2) ** (-101)


def read_file(filename, limit):
    zeros = []
    path = DATA / filename

    with path.open('rb') as f:
        number_of_blocks = struct.unpack('Q', f.read(8))[0]

        for _ in range(number_of_blocks):
            header = f.read(32)

            if len(header) < 32:
                break

            t0, t1, N0, N1 = struct.unpack('ddQQ', header)
            mpmath.mp.prec = int(log2(t1)) + 120
            t0 = mpmath.mpf(t0)
            Z = 0
            for _ in range(N1 - N0):
                z1, z2, z3 = struct.unpack('QIB', f.read(13))
                Z += (z3 << 96) + (z2 << 64) + z1
                gamma = t0 + mpmath.mpf(Z) * EPS
                zeros.append(float(gamma))
                if len(zeros) >= limit:
                    return zeros

    return zeros


all_zeros = []
file_counts = []

for filename in files:
    before_count = len(all_zeros)
    remaining = 1_000_000 - len(all_zeros)
    all_zeros.extend(read_file(filename, remaining))
    file_counts.append((filename, len(all_zeros) - before_count))

    if len(all_zeros) >= 1_000_000:
        break

np.save(ZEROS_OUTPUT, np.array(all_zeros))

TABLES_DIR.mkdir(parents=True, exist_ok=True)
summary_path = TABLES_DIR / 'extracted_zeros_summary.csv'
with summary_path.open('w', newline='', encoding='utf-8') as fh:
    writer = csv.writer(fh)
    writer.writerow(['source_file', 'zeros_loaded'])
    for source_file, count in file_counts:
        writer.writerow([source_file, count])
    writer.writerow(['TOTAL', len(all_zeros)])

preview_path = TABLES_DIR / 'extracted_zeros_preview.csv'
with preview_path.open('w', newline='', encoding='utf-8') as fh:
    writer = csv.writer(fh)
    writer.writerow(['index', 'gamma'])
    for index, gamma in enumerate(all_zeros[:5], start=1):
        writer.writerow([index, gamma])

print(f'Saved zeros array: {ZEROS_OUTPUT}')
print(f'Saved table: {summary_path}')
print(f'Saved table: {preview_path}')