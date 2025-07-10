import numpy as np
import scipy.integrate as integrate
from scipy import optimize
from multiprocessing import Pool
from functools import lru_cache
import dynesty

# === Model-Specific Functions ===

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
        (coeff_cd * x**2 + coeff_ab)
        / (Om_ab(x) * ((1 + x**2) * Om_ab(x) + 6 * Om_cd(x)**2))
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
        8 * a**2 * ((1 + x**2) * Om_ab(x) + 6 * Om_cd(x)**2)
        / (x**2 * Om_ab(x) * (x**2 * coeff_cd + coeff_ab))
    )

def eta(a, b, c, d):
    Om_ab = Om2(a, b)
    coeff_ab = coeff(a, c)
    coeff_cd = coeff(b, d)

    def eta_expr(x):
        x2 = x**2
        x4 = x2**2
        x6 = x2 * x4
        x8 = x4**2

        num = (
            4 * a * (
                3 * a * coeff_ab**2
                + 4 * coeff_ab * (b * (a - 3 * c**2) + a * (a + 12 * c * d + 3 * d**2)) * x2
                + (b * (7 * a**2 + 24 * a * c * (c + d) - 36 * c**2 * d * (2 * c + 3 * d))
                   - b**2 * (a + 24 * c**2)
                   + 12 * a * d**2 * (4 * a + 3 * c * (5 * c + 6 * d))) * x4
                - 2 * coeff_cd * (b**2 + 12 * b * c * d - 12 * a * d**2 - a * b) * x6
                - b * coeff_cd**2 * x8
            )
        )
        denom = x2 * Om_ab(x) * (coeff_ab + coeff_cd * x2)**2
        return num / denom

    return eta_expr

# === End of Inflation and COBE ===

def h_end(a, b, c, d):
    """Inflaton value where ε = 1"""
    f_eps = lambda x: eps(a, b, c, d)(x) - 1
    return optimize.brentq(f_eps, 1e-12, 100)

def h_in(a, b, c, d):
    """Inflaton value from COBE normalization"""
    V1 = V(a, b)
    epsi = eps(a, b, c, d)
    COBE = lambda x: V1(x) / epsi(x) - 5e-7
    return optimize.brentq(COBE, 1e-12, 1000.0)

def NEF(a, b, c, d):
    """Number of e-folds"""
    k1 = K(a, b, c, d)
    V1 = V(a, b)
    DV1 = DV(a, b)
    hi = h_in(a, b, c, d)
    he = h_end(a, b, c, d)
    integrand = lambda x: k1(x) * V1(x) / DV1(x)
    result, _ = integrate.quad(integrand, he, hi)
    return result

# === Likelihood and Prior ===

def loglike(x):
    a, b, c, d = [10**xi for xi in x]
    return -10000 * (55 - NEF(a, b, c, d))**2

def prior_transform(u):
    return np.array([
        6. * u[0] - 8.,
        16. * u[1] - 8.,
        16. * u[2] - 8.,
        16. * u[3] - 8.
    ])

# === Sampling ===

ndim = 4
np.random.seed(17)

if __name__ == "__main__":
    with Pool() as pool:
        sampler = dynesty.DynamicNestedSampler(
            loglike, prior_transform, ndim, nlive=1500,
            pool=pool, queue_size=8
        )
        sampler.run_nested(dlogz_init=0.5)

    
sresults = sampler.results
SAMPLE=sresults.samples
np.save('SAMPLE',SAMPLE)