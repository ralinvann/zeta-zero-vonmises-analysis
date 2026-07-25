import numpy as np
import csv
from pathlib import Path
from scipy import stats
from scipy.stats import vonmises

C   = 4.0
np.random.seed(42)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

N_BOOTSTRAP = 1000  # bootstrap resamples for SE estimation


def bootstrap_se(theta, n_boot=N_BOOTSTRAP, seed=0):
    """
    Estimate standard errors of von Mises MLE parameters (kappa, mu)
    via bootstrap resampling on the GUE control angle data.
    """
    rng = np.random.default_rng(seed)
    n   = len(theta)
    kappas, mus = [], []
    for _ in range(n_boot):
        resample = theta[rng.integers(0, n, size=n)]
        kappa, mu, _ = vonmises.fit(resample, fscale=1)
        kappas.append(kappa)
        mus.append(mu)
    return np.std(kappas), np.std(mus)


def sample_wigner_surmise(n_samples):
    x = stats.chi.rvs(df=3, size=n_samples)
    return x / np.sqrt(8 / np.pi)


def run_pipeline(spacings, C):
    theta = np.mod((spacings / C) * 2 * np.pi, 2 * np.pi)
    kappa, loc, _ = vonmises.fit(theta, fscale=1)
    return kappa, loc, theta


def save_summary_table(rows, output_path):
    headers = ["N", "mu", "se_mu", "kappa", "se_kappa", "cutoff"]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([
                row["N"],
                row["loc"],
                row["se_mu"],
                row["kappa"],
                row["se_kappa"],
                C,
            ])


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table_path = TABLES_DIR / "gue_kappa_summary.csv"

    n_values = [100, 1000, 10000, 100000, 1000000]
    gue_results = []

    for n_value in n_values:
        s = sample_wigner_surmise(n_value - 1)
        kappa, mu, theta = run_pipeline(s, C)
        se_kappa, se_mu = bootstrap_se(theta, seed=n_value)
        gue_results.append({
            "N": n_value,
            "kappa": kappa,
            "loc": mu,
            "se_kappa": se_kappa,
            "se_mu": se_mu,
        })

    save_summary_table(gue_results, table_path)
    print(f"Saved table: {table_path}")


if __name__ == "__main__":
    main()