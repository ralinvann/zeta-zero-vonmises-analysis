import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import vonmises

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'

N_BOOTSTRAP = 1000  # number of bootstrap resamples
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZEROS_PATH = PROJECT_ROOT / "zeros.npy"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def bootstrap_se(theta, n_boot=N_BOOTSTRAP, seed=42):
    """
    Estimate standard errors of von Mises MLE parameters (kappa, mu)
    via bootstrap resampling. Since zeta zeros are deterministic, these
    SEs reflect estimator sensitivity to the specific subset used, not
    true sampling variability.
    """
    rng = np.random.default_rng(seed)
    n = len(theta)
    kappas, mus = [], []
    for _ in range(n_boot):
        resample = theta[rng.integers(0, n, size=n)]
        kappa, mu, _ = vonmises.fit(resample, fscale=1)
        kappas.append(kappa)
        mus.append(mu)
    return np.std(kappas), np.std(mus)


def load_gammas(path):
    return np.load(path)


def analyze_sample(gammas, n_value, cutoff=4.0):
    raw_spacings = np.diff(gammas)
    T_mid  = (gammas[1:] + gammas[:-1]) / 2
    N_prime = (np.log(T_mid / (2 * np.pi)) + 1) / (2 * np.pi)
    unfolded = raw_spacings * N_prime
    theta    = np.mod((unfolded / cutoff) * 2 * np.pi, 2 * np.pi)

    kappa, loc, _ = vonmises.fit(theta, fscale=1)
    se_kappa, se_mu = bootstrap_se(theta)

    return {
        'N':       n_value,
        'theta':   theta,
        'kappa':   kappa,
        'loc':     loc,
        'se_kappa': se_kappa,
        'se_mu':    se_mu,
        'mean_raw': float(raw_spacings.mean()),
        'mean_unfolded': float(unfolded.mean()),
        'max_unfolded': float(unfolded.max()),
        'cutoff': float(cutoff),
    }


def plot_five_panel(results, output_dir):
    fig = plt.figure(figsize=(15, 9))
    gs  = fig.add_gridspec(2, 6)

    axes = [
        fig.add_subplot(gs[0, 0:2], projection='polar'),
        fig.add_subplot(gs[0, 2:4], projection='polar'),
        fig.add_subplot(gs[0, 4:6], projection='polar'),
        fig.add_subplot(gs[1, 1:3], projection='polar'),
        fig.add_subplot(gs[1, 3:5], projection='polar'),
    ]

    for ax, result in zip(axes, results):
        theta   = result['theta']
        kappa   = result['kappa']
        loc     = result['loc']
        n_value = result['N']
        se_k    = result['se_kappa']

        ax.hist(theta, bins=30, density=True,
                alpha=0.6, color='#4C72B0', edgecolor='white')
        x = np.linspace(0, 2 * np.pi, 400)
        ax.plot(x, vonmises.pdf(x, kappa, loc=loc),
                color='#DD8452', linewidth=2)

        exp = int(np.log10(n_value))
        # show kappa ± SE in each panel title
        ax.set_title(rf"$N=10^{exp}$,  $\hat{{\kappa}}={kappa:.3f} \pm {se_k:.3f}$",
                     pad=12, fontsize=10)

    fig.suptitle("von Mises fits to unfolded zeta zero spacings across sample sizes",
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = output_dir / "vonmises_fits_5panel.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_summary_table(results, output_path):
    headers = [
        "N",
        "mu",
        "se_mu",
        "kappa",
        "se_kappa",
        "mean_raw_spacing",
        "mean_unfolded_spacing",
        "max_unfolded_spacing",
        "cutoff",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for row in results:
            writer.writerow([
                row["N"],
                row["loc"],
                row["se_mu"],
                row["kappa"],
                row["se_kappa"],
                row["mean_raw"],
                row["mean_unfolded"],
                row["max_unfolded"],
                row["cutoff"],
            ])


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_gammas = load_gammas(ZEROS_PATH)
    results = []

    for n_value in [10 ** k for k in range(2, 7)]:
        if n_value > len(all_gammas):
            continue
        results.append(analyze_sample(all_gammas[:n_value], n_value))

    if not results:
        return

    plot_path = FIGURES_DIR / "vonmises_fits_5panel.png"
    table_path = TABLES_DIR / "real_zero_kappa_summary.csv"
    plot_five_panel(results, FIGURES_DIR)
    save_summary_table(results, table_path)
    print(f"Saved figure: {plot_path}")
    print(f"Saved table: {table_path}")


if __name__ == "__main__":
    main()