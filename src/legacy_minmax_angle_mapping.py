from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import vonmises

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZEROS_PATH = PROJECT_ROOT / "zeros.npy"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def legacy_minmax_angle_map(spacings):
    mean_spacing = float(np.mean(spacings))
    std_spacing = float(np.std(spacings))
    normalized_spacings = (spacings - mean_spacing) / std_spacing

    s_min = float(np.min(normalized_spacings))
    s_max = float(np.max(normalized_spacings))
    theta = 2 * np.pi * (normalized_spacings - s_min) / (s_max - s_min)
    return theta, mean_spacing, s_min, s_max


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    gammas = np.load(ZEROS_PATH)[:100000]
    spacings = np.diff(gammas)

    # Keep this legacy angle mapping for appendix comparison against the
    # current modulo-cutoff mapping used in the main analysis pipeline.
    theta, mean_spacing, s_min, s_max = legacy_minmax_angle_map(spacings)
    kappa, loc, _ = vonmises.fit(theta, fscale=1)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    ax.hist(theta, bins=30, density=True, alpha=0.6, color='#4C72B0', edgecolor='white')

    x = np.linspace(0, 2 * np.pi, 400)
    ax.plot(x, vonmises.pdf(x, kappa, loc=loc), color='#DD8452', linewidth=2)
    ax.set_title(r'Legacy min-max angle mapping fit ($N=10^5$)', fontsize=12)

    plt.tight_layout()
    figure_path = FIGURES_DIR / "legacy_minmax_vonmises_fit.png"
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Estimated mean direction mu: {loc}")
    print(f"Estimated concentration kappa: {kappa}")
    print(f"Mean spacing: {mean_spacing}")
    print(f"X minimum: {s_min}")
    print(f"X maximum: {s_max}")
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()