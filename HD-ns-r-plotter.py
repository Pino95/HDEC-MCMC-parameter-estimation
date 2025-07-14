import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import matplotlib.patches as mpatches
from tqdm import tqdm
from joblib import Parallel, delayed

# === Plot styling for LaTeX-quality output ===
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern}",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 18
})

# === Inflationary Model Functions ===


def coeff(j, k):
    return j + 6 * k**2

def Om2(a, b):
    return lambda x: a + b * x**2

def K(a, b, c, d):
    Om_ab = Om2(a, b)
    Om_cd = Om2(c, d)
    return lambda x: (
        (coeff(b, d) * x**2 + coeff(a, c)) /
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
    return lambda x: (
        8 * a**2 * ((1 + x**2) * Om_ab(x) + 6 * Om_cd(x)**2) /
        (x**2 * Om_ab(x) * (x**2 * coeff(b, d) + coeff(a, c)))
    )

def eta(a, b, c, d):
    Om_ab = Om2(a, b)
    coeff_ab = coeff(a, c)
    coeff_cd = coeff(b, d)
    return lambda x: (
        4 * a * (
            3 * a * coeff_ab**2 +
            4 * coeff_ab * (b * (a - 3 * c**2) + a * (a + 12 * c * d + 3 * d**2)) * x**2 +
            (b * (7 * a**2 + 24 * a * c * (c + d) - 36 * c**2 * d * (2 * c + 3 * d))
             - b**2 * (a + 24 * c**2) +
             12 * a * d**2 * (4 * a + 3 * c * (5 * c + 6 * d))) * x**4 -
            2 * coeff_cd * (b**2 + 12 * b * c * d - 12 * a * d**2 - a * b) * x**6 -
            b * coeff_cd**2 * x**8
        ) /
        (x**2 * Om_ab(x) * (coeff_ab + coeff_cd * x**2)**2)
    )

# === End of inflation and horizon-exit conditions ===
def h_end(a, b, c, d):
    try:
        epsi = optimize.brentq(lambda x: eps(a, b, c, d)(x) - 1, 1e-12, 1000)
        etai = optimize.brentq(lambda x: np.abs(eta(a, b, c, d)(x)) - 1, 1e-12, 1000)
        return max(epsi, etai)
    except ValueError:
        return np.nan

def h_in(a, b, c, d):
    try:
        V1 = V(a, b)
        eps1 = eps(a, b, c, d)
        return optimize.brentq(lambda x: V1(x) / eps1(x) - 5e-7, 1e-12, 10000)
    except ValueError:
        return np.nan

# === Observables ===
def n_s(a, b, c, d, x):
    return 1 - 6 * eps(a, b, c, d)(x) + 2 * eta(a, b, c, d)(x)

def r(a, b, c, d, x):
    return 16 * eps(a, b, c, d)(x)

def NEF(a, b, c, d, xi):
    try:
        he = h_end(a, b, c, d)
        if np.isnan(he) or np.isnan(xi):
            return np.nan
        k1 = K(a, b, c, d)
        DV1 = DV(a, b)
        V1 = V(a, b)
        integrand = lambda x: k1(x) * V1(x) / DV1(x)
        N, _ = integrate.quad(integrand, he, xi)
        return N
    except Exception:
        return np.nan

# === Load parameter data ===
par_data = np.load('SAMPLE.npy')
par0, par1, par2, par3 = par_data[:, 0], par_data[:, 1], par_data[:, 2], par_data[:, 3]
Ns = np.load('NS.npy')
R = np.load('R.npy')

# === Load observational contours ===
bcp1 = np.loadtxt('bck1s.txt')
bcp2 = np.loadtxt('bck2s.txt')
ACT1 = np.loadtxt('ACT1s.txt')
ACT2 = np.loadtxt('ACT2s.txt')

# === Extract observational bands ===
NS1s, R1s = bcp1[:, 0], bcp1[:, 1]
NS2s, R2s = bcp2[:, 0], bcp2[:, 1]
NS_ACT1, R_ACT1 = ACT1[:, 0], ACT1[:, 1]
NS_ACT2, R_ACT2 = ACT2[:, 0], ACT2[:, 1]

# === Parallel NEF computation ===
def compute_nef_for_index(i):
    a, b, c, d = 10**par0[i], 10**par1[i], 10**par2[i], 10**par3[i]
    hi = h_in(a, b, c, d)
    return NEF(a, b, c, d, hi)

nef_vals = np.array(
    Parallel(n_jobs=-1)(
        delayed(compute_nef_for_index)(i) for i in tqdm(range(len(par0)), desc="Computing NEF")
    )
)

# === Mask samples with 54 < N < 56 ===
mask = (nef_vals > 54) & (nef_vals < 56)
Ns_masked = Ns[mask]
R_masked = R[mask]
PAR_masked = par2[mask]  # color = log10 tau_eta

# === Plotting ===
fig, ax = plt.subplots(figsize=(8, 6))

# 1σ Planck region
ax.plot(NS1s, np.log10(R1s), color="green", label=r"1 $\sigma$", linewidth=1)
ax.fill_betweenx(np.log10(R1s), NS1s, alpha=0.07, color="green")

# 1σ ACT region
ax.plot(NS_ACT1, np.log10(R_ACT1), color="purple", label="ACT 1 $\sigma$", linewidth=1)
ax.fill_betweenx(np.log10(R_ACT1), NS_ACT1, alpha=0.07, color="purple")

# Data points
img = ax.scatter(Ns_masked, np.log10(R_masked), c=PAR_masked, cmap='Blues', marker=".", s=2, vmin=-8, vmax=8)

# Legend patches
planck_patch = mpatches.Patch(color='lightgreen', alpha=0.2, label='Planck+BK18')
act_patch = mpatches.Patch(color='violet', alpha=0.2, label='Planck+BK18+ACT+LB')
legend = ax.legend(handles=[planck_patch, act_patch], loc='upper right')
legend.set_frame_on(False)

# Axes configuration
ax.set_xlim([0.955, 0.985])
ax.set_ylim([-11, 0.5])
ax.set_xlabel(r'$n_s$')
ax.set_ylabel(r'$\log_{10}\;r$')
fig.colorbar(img, orientation="vertical", pad=0.07, label=r'$\log_{10}\;\tau_\eta$')

# Save plot
plt.savefig('PLANCK-ACT-NS-R-taueta_contraints.png', dpi=500, bbox_inches='tight')

    

    
    
