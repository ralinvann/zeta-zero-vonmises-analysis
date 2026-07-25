import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from scipy.stats import vonmises

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZEROS_PATH = PROJECT_ROOT / "zeros.npy"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

gammas = np.load(ZEROS_PATH)[:1000]

raw_spacings = np.diff(gammas)
T_mid = (gammas[1:] + gammas[:-1]) / 2
N_prime = (np.log(T_mid / (2 * np.pi)) + 1) / (2 * np.pi)
unfolded = raw_spacings * N_prime

C_values = np.arange(1.0, 6.1, 0.2)
kappas = []
for C in C_values:
    theta = (unfolded / C) * 2 * np.pi
    theta = np.mod(theta, 2 * np.pi)
    kappa, loc, scale = vonmises.fit(theta, fscale=1)
    kappas.append(kappa)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(C_values, kappas, color='#4C72B0', linewidth=2, marker='o', markersize=4)
ax.axvline(4.0, color='#DD8452', linestyle='--', linewidth=1.5, label='Chosen $C=4$')
ax.axvspan(2.68, 6.1, alpha=0.08, color='#55A868', label='No wraparound region ($C >$ observed max)')
ax.set_xlabel(r'Cutoff parameter $C$', fontsize=12)
ax.set_ylabel(r'Estimated concentration $\kappa$', fontsize=12)
ax.set_title(r'Sensitivity of $\kappa$ to choice of cutoff $C$ ($N=1000$)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

figure_path = FIGURES_DIR / "kappa_vs_C.png"
table_path = TABLES_DIR / "kappa_vs_cutoff.csv"

plt.savefig(figure_path, dpi=300, bbox_inches='tight')

with table_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["cutoff_C", "kappa"])
    for c_value, kappa in zip(C_values, kappas):
        writer.writerow([float(c_value), float(kappa)])

print(f"Saved figure: {figure_path}")
print(f"Saved table: {table_path}")
