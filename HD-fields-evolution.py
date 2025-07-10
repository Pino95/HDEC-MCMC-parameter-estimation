import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

# === Parameters === Careful about the Planck mass. Here we work in Planck units, if you use something different be consistent once asked initial conditions
MAX_EFOLDS = 20000
M_P = 1.0
xi_h = 5000
xi_chi = 0.001
lambda_ = 0.001
alpha = 0

# === Omega^2 === Conformal factor
def omega2(h, chi):
    return (xi_h * h**2 + xi_chi * chi**2) / M_P**2

# === Potential and Derivatives === You can compute them numerically, but having them analytical is much more stable for long time evolutions
def V_E(h, chi):
    num = lambda_ * (h**2 - alpha * chi**2)**2
    denom = (xi_h * h**2 + xi_chi * chi**2)**2
    return num / denom

def dV_dh(h, chi):
    num = 4 * lambda_ * h * (h**2 - alpha * chi**2)
    denom = (xi_h * h**2 + xi_chi * chi**2)**2
    term = 4 * lambda_ * (h**2 - alpha * chi**2)**2 * xi_h * h
    return (num * (xi_h * h**2 + xi_chi * chi**2) - term) / (denom**2)

def dV_dchi(h, chi):
    num = -4 * lambda_ * alpha * chi * (h**2 - alpha * chi**2)
    denom = (xi_h * h**2 + xi_chi * chi**2)**2
    term = 4 * lambda_ * (h**2 - alpha * chi**2)**2 * xi_chi * chi
    return (num * (xi_h * h**2 + xi_chi * chi**2) - term) / (denom**2)

# === Field-space metric === You can compute them numerically, but having them analytical is much more stable for long time evolutions
def G_metric(h, chi):
    Om2 = omega2(h, chi)
    Om4 = Om2**2
    G = np.zeros((2, 2))
    G[0, 0] = 1 / Om2 + (6 * xi_h**2 * h**2) / (M_P**2 * Om4)
    G[1, 1] = 1 / Om2 + (6 * xi_chi**2 * chi**2) / (M_P**2 * Om4)
    G[0, 1] = G[1, 0] = (6 * xi_h * xi_chi * h * chi) / (M_P**2 * Om4)
    return G

def G_inv(h, chi):
    G = G_metric(h, chi)
    return np.linalg.inv(G)

def G_metric_derivatives(h, chi):
    eps = 1e-5
    dG_dh = (G_metric(h + eps, chi) - G_metric(h - eps, chi)) / (2 * eps)
    dG_dchi = (G_metric(h, chi + eps) - G_metric(h, chi - eps)) / (2 * eps)
    return dG_dh, dG_dchi

# ATTENTION. This are the christoffel symbols for the field-space metric, not the usual one you compute in GR
def christoffel_symbols(h, chi):
    Ginv = G_inv(h, chi)
    dG_dh, dG_dchi = G_metric_derivatives(h, chi)

    Gamma = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                dG_jk_h = dG_dh[j, k]
                dG_kj_h = dG_dh[k, j]
                dG_jk_chi = dG_dchi[j, k]
                dG_kj_chi = dG_dchi[k, j]

                if i == 0:
                    term = dG_jk_h + dG_kj_h - dG_dh[j, k]
                else:
                    term = dG_jk_chi + dG_kj_chi - dG_dchi[j, k]

                for l in range(2):
                    Gamma[i, j, k] += 0.5 * Ginv[i, l] * term
    return Gamma

def kinetic_energy(h, chi, hdot, chidot):
    G = G_metric(h, chi)
    v = np.array([hdot, chidot])
    return 0.5 * v @ G @ v

def Hubble(h, chi, hdot, chidot):
    V = V_E(h, chi)
    KE = kinetic_energy(h, chi, hdot, chidot)
    return np.sqrt((KE + V) / (3 * M_P**2))

def background_eom_efolds(N, y):
    h, chi, hdot, chidot = y
    Ginv = G_inv(h, chi)
    dV = np.array([dV_dh(h, chi), dV_dchi(h, chi)])
    pi = np.array([hdot, chidot])
    H = Hubble(h, chi, hdot, chidot)
    Gamma = christoffel_symbols(h, chi)

    dN_hdot = (-3 * hdot - Ginv[0] @ dV / H)
    dN_chidot = (-3 * chidot - Ginv[1] @ dV / H)
    for j in range(2):
        for k in range(2):
            dN_hdot -= Gamma[0, j, k] * pi[j] * pi[k] / H
            dN_chidot -= Gamma[1, j, k] * pi[j] * pi[k] / H

    return [hdot / H, chidot / H, dN_hdot, dN_chidot]

def epsilon_slowroll(h, chi, hdot, chidot):
    H = Hubble(h, chi, hdot, chidot)
    KE = kinetic_energy(h, chi, hdot, chidot)
    return KE / (H**2 * M_P**2)

def end_of_inflation(N, y):
    h, chi, hdot, chidot = y
    epsilon = epsilon_slowroll(h, chi, hdot, chidot)
    return epsilon - 1.0
end_of_inflation.terminal = True
end_of_inflation.direction = 1

def run_simulation(h0, chi0, hdot0, chidot0):
    y0 = [h0, chi0, hdot0, chidot0]
    N_eval = np.linspace(0, MAX_EFOLDS, int(10 * MAX_EFOLDS))
    try:
        sol = solve_ivp(
            background_eom_efolds,
            (0, MAX_EFOLDS),
            y0,
            t_eval=N_eval,
            rtol=1e-8,
            atol=1e-10,
            events=end_of_inflation
        )
        return (h0, chi0, sol)
    except Exception as e:
        print(f"Simulation failed: h0={h0}, chi0={chi0}, Error: {e}")
        return (h0, chi0, None)

def simulate_from_tuple(args):
    return run_simulation(*args)

def main():
    init_conds = []
    n_runs = int(input("Enter number of simulations: "))
    for i in range(n_runs):
        h0 = float(input(f"Enter h0 for run {i+1}: "))
        chi0 = float(input(f"Enter chi0 for run {i+1}: "))
        hdot0 = float(input(f"Enter hdot0 for run {i+1}: "))
        chidot0 = float(input(f"Enter chidot0 for run {i+1}: "))
        init_conds.append((h0, chi0, hdot0, chidot0))
    colors = ['blue', 'green', 'orange', 'purple']

    results = [simulate_from_tuple(ic) for ic in init_conds]

    print("All simulations complete.")
    for h0, chi0, sol in results:
        if sol is not None:
            print(f"Final e-folds for h0={h0}, chi0={chi0}: N = {sol.t[-1]:.2f}")

    # === Slow-roll parameter epsilon ===
    plt.figure(figsize=(8, 6))
    for idx, (h0, chi0, sol) in enumerate(results):
        if sol is not None:
            epsilons = []
            for i in range(len(sol.t)):
                h, chi, hdot, chidot = sol.y[:, i]
                epsilon = epsilon_slowroll(h, chi, hdot, chidot)
                epsilons.append(epsilon)
            plt.plot(sol.t, epsilons, label=f'h0={h0}, chi0={chi0}', color=colors[idx])
    plt.xlabel('N (e-folds)')
    plt.ylabel('epsilon')
    plt.title('Slow-roll Parameter ε Over E-folds')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Plot field space trajectory ===
    plt.figure(figsize=(8, 6))
    for idx, (h0, chi0, sol) in enumerate(results):
        if sol is not None:
            plt.plot(sol.y[1], sol.y[0], label=f'h0={h0}, chi0={chi0}', color=colors[idx])
            plt.plot(sol.y[1][-1], sol.y[0][-1], 'o', color=colors[idx], markersize=6, label=f'end h0={h0}, chi0={chi0}')
    plt.xlabel('chi')
    plt.ylabel('h')
    plt.title('Field Trajectories')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()