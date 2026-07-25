# Riemann Zero Spacing Analysis

This project analyzes spacings of nontrivial Riemann zeta zeros and fits circular statistics (von Mises models) to transformed spacing data.

## Directory layout

- `data/`: source and auxiliary data files
- `src/`: analysis scripts
- `results/figures/`: generated plots
- `results/tables/`: generated CSV tables

## Quick start

1. Create and activate a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run scripts in `src/` as needed.

Example:

```bash
python src/extract_riemann_zeros.py
python src/analyze_real_spacings.py
python src/analyze_unfolded_spacings.py
python src/vonmises_fit.py
python src/diagnostics.py
```
