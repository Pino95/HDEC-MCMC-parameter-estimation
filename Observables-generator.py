import numpy as np
from scipy import optimize
from joblib import Parallel, delayed
from functools import lru_cache
import logging
import time
from tqdm import tqdm

# === Logging setup ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Model Functions ===

@lru_cache(None)
def coeff(j, k):
    return j + 6 * k**2

def Om2(a, b):
    return lambda x: a + b * x**2

def K(a, b, c, d):
    Om_ab = Om2(a, b)
    Om_cd = Om2(c, d)
    coeff_ab = coeff(a, c)
    coeff_cd = coeff(b, d)
    return lambda x: (
        (coeff_cd * x**2 + coeff_ab) /
        (Om_ab(x) * ((1 + x**2) * Om_ab(x) + 6 * Om_cd(x)**2))
    )

def V(a, b):
    Om_ab = Om2(a, b)
    return lambda x: (0.001 * x**4) / (4 * Om_ab(x)**2)

def DV(a, b):
    Om_ab = Om2(a, b)
    return lambda x: (0.001 * a * x**3) / (Om_ab(x)**3)

def eps(a, b, c, d):
    Om_ab = Om2(a, b)
    Om_cd = Om2(c, d)
    coeff_ab = coeff(a, c)
    coeff_cd = coeff(b, d)
    return lambda x: (
        8 * a**2 * ((1 + x**2) * Om_ab(x) + 6 * Om_cd(x)**2) /
        (x**2 * Om_ab(x) * (x**2 * coeff_cd + coeff_ab))
    )

def eta(a, b, c, d):
    Om_ab = Om2(a, b)
    coeff_ab = coeff(a, c)
    coeff_cd = coeff(b, d)

    def eta_func(x):
        x2 = x**2
        x4 = x2**2
        x6 = x2 * x4
        x8 = x4**2
        denom = x2 * Om_ab(x) * (coeff_ab + coeff_cd * x2)**2
        if denom == 0:
            return np.nan

        num = (
            4 * a * (
                3 * a * coeff_ab**2 +
                4 * coeff_ab * (b * (a - 3 * c**2) + a * (a + 12 * c * d + 3 * d**2)) * x2 +
                (b * (7 * a**2 + 24 * a * c * (c + d) - 36 * c**2 * d * (2 * c + 3 * d))
                 - b**2 * (a + 24 * c**2) +
                 12 * a * d**2 * (4 * a + 3 * c * (5 * c + 6 * d))) * x4 -
                2 * coeff_cd * (b**2 + 12 * b * c * d - 12 * a * d**2 - a * b) * x6 -
                b * coeff_cd**2 * x8
            )
        )
        return num / denom

    return eta_func

# === Root Finding Utilities ===

def h_end(a, b, c, d):
    try:
        return optimize.brentq(lambda x: eps(a, b, c, d)(x) - 1, 1e-12, 100)
    except ValueError:
        return np.nan

def h_in(a, b, c, d):
    try:
        V1 = V(a, b)
        eps1 = eps(a, b, c, d)
        return optimize.brentq(lambda x: V1(x) / eps1(x) - 5e-7, 1e-12, 1000)
    except ValueError:
        return np.nan

# === Observables ===

def n_s(a, b, c, d, x):
    ep = eps(a, b, c, d)
    et = eta(a, b, c, d)
    return 1 - 6 * ep(x) + 2 * et(x)

def r(a, b, c, d, x):
    return 16 * eps(a, b, c, d)(x)

# === Parallelized Computation ===

def process_sample(j, par0, par1, par2, par3):
    try:
        a, b, c, d = 10**par0[j], 10**par1[j], 10**par2[j], 10**par3[j]
        x = h_in(a, b, c, d)
        if np.isnan(x):
            raise ValueError("Invalid inflaton value")
        ns_val = n_s(a, b, c, d, x)
        r_val = r(a, b, c, d, x)
        return (x, ns_val, r_val)
    except Exception as e:
        logging.warning(f"[Sample {j}] Failed: {e}")
        return (np.nan, np.nan, np.nan)

# === Main Execution ===

def main():
    start_time = time.time()

    try:
        data = np.load('SAMPLE.npy')
    except FileNotFoundError:
        logging.error("File 'SAMPLE.npy' not found.")
        return

    if data.shape[1] != 4:
        logging.error("Expected data shape (N, 4).")
        return

    par0, par1, par2, par3 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    N = len(par0)

    logging.info(f"Processing {N} samples in parallel...")

    results = Parallel(n_jobs=-1)(
        delayed(process_sample)(j, par0, par1, par2, par3)
        for j in tqdm(range(N), desc="Evaluating Samples")
    )

    xin, Ns, R = zip(*results)

    np.save('NS.npy', np.array(Ns))
    np.save('R.npy', np.array(R))

    elapsed = time.time() - start_time
    logging.info(f"Completed {N} samples in {elapsed:.2f} seconds.")

if __name__ == '__main__':
    main()


    