import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------
# GLOBAL DEFAULTS (for sliders etc.)
# ---------------------------------------------------------------------

CHI_DEFAULT = 2.0     # baseline chi for nondrastic case
PSI_DEFAULT = 0.02    # discovery probability per person
EPS_DEFAULT = 0.01    # fraction of disappearing varieties


# ---------------------------------------------------------------------
# 1. CORE FUNCTIONS: PROFIT, INNOVATION, GROWTH
# ---------------------------------------------------------------------

def pi_drastic(alpha):
    """Profit parameter pi in drastic one-sector model."""
    return (1.0 - alpha) * alpha ** ((1.0 + alpha) / (1.0 - alpha))


def pi_nondrastic(alpha, chi):
    """Profit parameter pi(chi) in nondrastic case."""
    return (chi - 1.0) * (alpha / chi) ** (1.0 / (1.0 - alpha))


def equilibrium_n(lambd, sigma, pi_term):
    """Solve phi'(n) * pi_term = 1 for phi(n) = lambd * n**sigma."""
    exponent = 1.0 / (sigma - 1.0)
    base = 1.0 / (lambd * sigma * pi_term)
    return base ** exponent


def innovation_probability(lambd, sigma, pi_term):
    """Innovation probability mu = phi(n)."""
    n = equilibrium_n(lambd, sigma, pi_term)
    return lambd * n ** sigma


def long_run_growth(lambd, sigma, pi_term, gamma):
    """Long-run growth rate g = mu * (gamma - 1)."""
    mu = innovation_probability(lambd, sigma, pi_term)
    return mu * (gamma - 1.0)


# ---------------------------------------------------------------------
# 2. ONE-SECTOR SIMULATION
# ---------------------------------------------------------------------

def simulate_one_sector(A0, T, alpha, gamma, lambd, sigma, L, pi_term, seed=0):
    """
    One-sector quality-ladder model.

    A_{t+1} = gamma * A_t with probability mu,
              A_t        otherwise.

    Y_t, GDP_t and profits are proportional to A_t * L.
    """
    rng = np.random.default_rng(seed)
    N = T
    mu = innovation_probability(lambd, sigma, pi_term)

    A = np.empty(N + 1)
    Y = np.empty(N + 1)
    GDP = np.empty(N + 1)
    g = np.empty(N)
    Pi = np.empty(N + 1)

    A[0] = A0

    y_factor = alpha ** (2.0 * alpha / (1.0 - alpha))
    x_factor = alpha ** (2.0 / (1.0 - alpha))
    pi_value = pi_term / L

    for t in range(N):
        if rng.uniform() < mu:
            A[t + 1] = gamma * A[t]
        else:
            A[t + 1] = A[t]
        g[t] = (A[t + 1] - A[t]) / A[t]

    for t in range(N + 1):
        Y[t] = y_factor * A[t] * L
        x_t = x_factor * A[t] * L
        GDP[t] = Y[t] - x_t
        Pi[t] = pi_value * A[t] * L

    time = np.arange(N + 1)

    return {
        "time": time,
        "A": A,
        "Y": Y,
        "GDP": GDP,
        "g": g,
        "Pi": Pi,
        "mu": mu,
    }


def simulate_drastic_one_sector(A0, T, alpha, gamma, lambd, sigma, L, seed=0):
    pi_val = pi_drastic(alpha)
    pi_term = pi_val * L
    return simulate_one_sector(A0, T, alpha, gamma, lambd, sigma, L, pi_term, seed=seed)


def simulate_nondrastic_one_sector(A0, T, alpha, gamma, lambd, sigma, L, chi, seed=0):
    pi_val = pi_nondrastic(alpha, chi)
    pi_term = pi_val * L
    return simulate_one_sector(A0, T, alpha, gamma, lambd, sigma, L, pi_term, seed=seed)


# ---------------------------------------------------------------------
# 3. MULTISECTOR MODEL
# ---------------------------------------------------------------------

def simulate_multisector(A0, T, alpha, gamma, lambd, sigma, L,
                         n_sectors=200, seed=0):
    """
    Multisector model with n_sectors independent ladders.

    In each sector i:
      A_{i,t+1} = gamma * A_{i,t} with probability mu,
                  A_{i,t}        otherwise.

    Aggregate A_t is cross-sector mean.
    """
    rng = np.random.default_rng(seed)
    N = T

    pi_val = pi_drastic(alpha)
    pi_term = pi_val * L
    mu = innovation_probability(lambd, sigma, pi_term)

    A_sectors = np.full((N + 1, n_sectors), A0)
    A_agg = np.empty(N + 1)
    Y_agg = np.empty(N + 1)
    GDP_agg = np.empty(N + 1)
    g_agg = np.empty(N)

    y_factor = alpha ** (2.0 * alpha / (1.0 - alpha))
    x_factor = alpha ** (2.0 / (1.0 - alpha))

    for t in range(N):
        A_current = A_sectors[t]
        shocks = rng.uniform(size=n_sectors) < mu
        A_next = A_current * np.where(shocks, gamma, 1.0)
        A_sectors[t + 1] = A_next

        A_agg[t] = A_current.mean()
        A_agg_next = A_next.mean()
        g_agg[t] = (A_agg_next - A_agg[t]) / A_agg[t]

    A_agg[N] = A_sectors[N].mean()

    for t in range(N + 1):
        Y_agg[t] = y_factor * A_agg[t] * L
        X_agg = x_factor * A_agg[t] * L
        GDP_agg[t] = Y_agg[t] - X_agg

    time = np.arange(N + 1)
    return {
        "time": time,
        "A_sectors": A_sectors,
        "A_agg": A_agg,
        "Y_agg": Y_agg,
        "GDP_agg": GDP_agg,
        "g_agg": g_agg,
        "mu": mu,
    }


# ---------------------------------------------------------------------
# 4. SCALE EFFECTS
# ---------------------------------------------------------------------

def simulate_M_path(L_value, psi, eps, T, M0):
    """M_{t+1} - M_t = psi * L - eps * M_t."""
    N = T
    M = np.empty(N + 1)
    M[0] = M0
    for t in range(N):
        M[t + 1] = M[t] + psi * L_value - eps * M[t]
    time = np.arange(N + 1)
    return time, M


def growth_with_scale_effects(L_values, alpha, gamma, lambd, sigma):
    """g(L) in one-sector model with scale effects."""
    pi_val = pi_drastic(alpha)
    g_vals = np.empty_like(L_values, dtype=float)
    for i, L_val in enumerate(L_values):
        pi_term = pi_val * L_val
        g_vals[i] = long_run_growth(lambd, sigma, pi_term, gamma)
    return g_vals


def growth_without_scale_effects(alpha, gamma, lambd, sigma, psi, eps):
    """g in variable-variety model (profit term proportional to eps/psi)."""
    pi_val = pi_drastic(alpha)
    pi_term = pi_val * (eps / psi)
    return long_run_growth(lambd, sigma, pi_term, gamma)


# ---------------------------------------------------------------------
# 5. STREAMLIT UI
# ---------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Aghion-Howitt Model Simulator", layout="wide")

    st.title("Aghion-Howitt Model Simulator")

    st.sidebar.header("Baseline parameters")

    alpha = st.sidebar.slider("alpha", 0.1, 0.7, 0.3, 0.01)
    gamma = st.sidebar.slider("gamma", 1.01, 1.5, 1.2, 0.01)
    lambd = st.sidebar.slider("lambda (research productivity)", 0.1, 1.0, 0.5, 0.05)
    sigma = st.sidebar.slider("sigma (elasticity in phi)", 0.1, 0.9, 0.5, 0.05)
    L = st.sidebar.slider("Population L", 0.5, 5.0, 1.0, 0.1)
    T = st.sidebar.slider("Number of periods T", 50, 400, 200, 10)
    A0 = st.sidebar.number_input("Initial productivity A0", 0.1, 5.0, 1.0, 0.1)
    seed = st.sidebar.number_input("Random seed", 0, 9999, 123, 1)

    mode = st.sidebar.radio(
        "Model version",
        [
            "One-sector: drastic innovations",
            "One-sector: nondrastic innovations",
            "Multisector model",
            "Scale effects",
        ],
    )

    # -------------------------------------------------------------
    # One-sector: drastic innovations
    # -------------------------------------------------------------
    if mode == "One-sector: drastic innovations":
        st.header("One-sector model with drastic innovations")

        sim_d = simulate_drastic_one_sector(A0, T, alpha, gamma, lambd, sigma, L, seed=int(seed))
        pi_val = pi_drastic(alpha)
        pi_term = pi_val * L
        mu_theory = innovation_probability(lambd, sigma, pi_term)
        g_theory = long_run_growth(lambd, sigma, pi_term, gamma)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Theoretical values**")
            st.write(f"pi = {pi_val:.4f}")
            st.write(f"mu (theoretical) = {mu_theory:.4f}")
            st.write(f"g (theoretical) = {g_theory:.4f}")
        with col2:
            st.markdown("**Simulation**")
            st.write(f"mu (simulation) = {sim_d['mu']:.4f}")
            st.write(f"mean g_t = {sim_d['g'].mean():.4f}")

        fig1, ax1 = plt.subplots()
        ax1.plot(sim_d["time"], sim_d["A"], label="A_t")
        ax1.plot(sim_d["time"], sim_d["Y"], label="Y_t")
        ax1.plot(sim_d["time"], sim_d["GDP"], label="GDP_t")
        ax1.set_xlabel("time")
        ax1.set_ylabel("levels")
        ax1.set_title("Drastic one-sector: A_t, Y_t, GDP_t")
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(sim_d["time"], sim_d["Pi"])
        ax2.set_xlabel("time")
        ax2.set_ylabel("Pi_t")
        ax2.set_title("Monopoly profits Pi_t")
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        ax3.hist(sim_d["g"], bins=30, edgecolor="black")
        ax3.set_xlabel("g_t")
        ax3.set_ylabel("frequency")
        ax3.set_title("Distribution of realised growth rates g_t")
        st.pyplot(fig3)

        st.subheader("Sensitivity of g to lambda, gamma, alpha")

        lambda_grid = np.linspace(0.2, 0.8, 15)
        gamma_grid = np.linspace(1.05, 1.4, 15)
        alpha_grid = np.linspace(0.1, 0.5, 21)

        g_lambda = [long_run_growth(lmb, sigma, pi_term, gamma) for lmb in lambda_grid]
        g_gamma = [long_run_growth(lambd, sigma, pi_term, gm) for gm in gamma_grid]

        g_alpha = []
        for a in alpha_grid:
            pi_a = pi_drastic(a)
            pi_term_a = pi_a * L
            g_alpha.append(long_run_growth(lambd, sigma, pi_term_a, gamma))

        fig4, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].plot(lambda_grid, g_lambda)
        axes[0].set_xlabel("lambda")
        axes[0].set_ylabel("g")
        axes[0].set_title("g(lambda)")

        axes[1].plot(gamma_grid, g_gamma)
        axes[1].set_xlabel("gamma")
        axes[1].set_title("g(gamma)")

        axes[2].plot(alpha_grid, g_alpha)
        axes[2].set_xlabel("alpha")
        axes[2].set_title("g(alpha)")

        fig4.tight_layout()
        st.pyplot(fig4)

    # -------------------------------------------------------------
    # One-sector: nondrastic innovations
    # -------------------------------------------------------------
    elif mode == "One-sector: nondrastic innovations":
        st.header("One-sector model with nondrastic innovations")

        chi = st.sidebar.slider("chi (limit price)", 1.05, 4.0, CHI_DEFAULT, 0.05)

        sim_nd = simulate_nondrastic_one_sector(
            A0, T, alpha, gamma, lambd, sigma, L, chi, seed=int(seed)
        )
        sim_d = simulate_drastic_one_sector(
            A0, T, alpha, gamma, lambd, sigma, L, seed=int(seed)
        )

        chi_grid = np.linspace(1.05, 4.0, 40)
        pi_chi = []
        g_chi = []
        mu_chi = []

        for chi_val in chi_grid:
            pi_val_chi = pi_nondrastic(alpha, chi_val)
            pi_term_chi = pi_val_chi * L
            mu_val = innovation_probability(lambd, sigma, pi_term_chi)
            g_val = mu_val * (gamma - 1.0)
            pi_chi.append(pi_val_chi)
            g_chi.append(g_val)
            mu_chi.append(mu_val)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Baseline nondrastic case**")
            st.write(f"chi = {chi:.3f}")
            st.write(f"mu (simulation) = {sim_nd['mu']:.4f}")
            st.write(f"mean g_t = {sim_nd['g'].mean():.4f}")
        with col2:
            st.markdown("**Drastic benchmark**")
            pi_d = pi_drastic(alpha)
            pi_term_d = pi_d * L
            mu_d = innovation_probability(lambd, sigma, pi_term_d)
            g_d = long_run_growth(lambd, sigma, pi_term_d, gamma)
            st.write(f"mu (drastic) = {mu_d:.4f}")
            st.write(f"g (drastic) = {g_d:.4f}")

        fig1, axes = plt.subplots(3, 1, figsize=(7, 8))
        axes[0].plot(chi_grid, pi_chi)
        axes[0].set_xlabel("chi")
        axes[0].set_ylabel("pi(chi)")
        axes[0].set_title("Profit parameter pi(chi)")

        axes[1].plot(chi_grid, mu_chi)
        axes[1].set_xlabel("chi")
        axes[1].set_ylabel("mu(chi)")
        axes[1].set_title("Innovation probability mu(chi)")

        axes[2].plot(chi_grid, g_chi)
        axes[2].set_xlabel("chi")
        axes[2].set_ylabel("g(chi)")
        axes[2].set_title("Long-run growth g(chi)")

        fig1.tight_layout()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(sim_d["time"], sim_d["GDP"], linestyle="--", label="drastic GDP_t")
        ax2.plot(sim_nd["time"], sim_nd["GDP"], label="nondrastic GDP_t")
        ax2.set_xlabel("time")
        ax2.set_ylabel("GDP_t")
        ax2.set_title("GDP: drastic vs nondrastic")
        ax2.legend()
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        ax3.plot(sim_nd["time"], sim_nd["Pi"], label="nondrastic Pi_t")
        ax3.plot(sim_d["time"], sim_d["Pi"], linestyle="--", label="drastic Pi_t")
        ax3.set_xlabel("time")
        ax3.set_ylabel("Pi_t")
        ax3.set_title("Profits: drastic vs nondrastic")
        ax3.legend()
        st.pyplot(fig3)

    # -------------------------------------------------------------
    # Multisector model
    # -------------------------------------------------------------
    elif mode == "Multisector model":
        st.header("Multisector model and smoothing of growth")

        n_sectors = st.sidebar.slider("Number of sectors", 10, 500, 200, 10)

        sim_d = simulate_drastic_one_sector(A0, T, alpha, gamma, lambd, sigma, L, seed=int(seed))
        sim_m = simulate_multisector(
            A0, T, alpha, gamma, lambd, sigma, L,
            n_sectors=n_sectors, seed=int(seed)
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Innovation probability**")
            st.write(f"mu (common) = {sim_m['mu']:.4f}")
        with col2:
            st.markdown("**Average growth rates**")
            st.write(f"mean g_t (one-sector) = {sim_d['g'].mean():.4f}")
            st.write(f"mean g_t (multisector) = {sim_m['g_agg'].mean():.4f}")

        fig1, ax1 = plt.subplots()
        ax1.plot(sim_m["time"], sim_m["A_agg"], label="A_t aggregate")
        ax1.plot(sim_m["time"], sim_m["Y_agg"], label="Y_t aggregate")
        ax1.plot(sim_m["time"], sim_m["GDP_agg"], label="GDP_t aggregate")
        ax1.set_xlabel("time")
        ax1.set_ylabel("levels")
        ax1.set_title("Multisector: aggregate variables")
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        for i in range(min(5, n_sectors)):
            ax2.plot(sim_m["time"], sim_m["A_sectors"][:, i])
        ax2.set_xlabel("time")
        ax2.set_ylabel("A_it")
        ax2.set_title("Selected sectoral productivity paths")
        st.pyplot(fig2)

        Tg = min(len(sim_d["g"]), len(sim_m["g_agg"]))
        time_g = np.arange(Tg)

        fig3, ax3 = plt.subplots()
        ax3.plot(time_g, sim_d["g"][:Tg], label="one-sector g_t")
        ax3.plot(time_g, sim_m["g_agg"][:Tg], linestyle="--", label="multisector g_t")
        ax3.set_xlabel("time")
        ax3.set_ylabel("g_t")
        ax3.set_title("Growth rates: one-sector vs multisector")
        ax3.legend()
        st.pyplot(fig3)

    # -------------------------------------------------------------
    # Scale effects
    # -------------------------------------------------------------
    elif mode == "Scale effects":
        st.header("Scale effects and population size")

        # psi, eps as user-controlled parameters
        psi = st.sidebar.slider("psi (discovery prob per person)", 0.001, 0.1, PSI_DEFAULT, 0.001)
        eps = st.sidebar.slider("eps (disappearance fraction)", 0.001, 0.1, EPS_DEFAULT, 0.001)

        L_min = st.sidebar.number_input("L grid min", 0.2, 5.0, 0.5, 0.1)
        L_max = st.sidebar.number_input("L grid max", 0.5, 20.0, 5.0, 0.5)
        grid_points = st.sidebar.slider("Grid points for L", 5, 50, 20, 1)
        L_values = np.linspace(L_min, L_max, grid_points)

        g_scale = growth_with_scale_effects(L_values, alpha, gamma, lambd, sigma)
        g_noscale = growth_without_scale_effects(alpha, gamma, lambd, sigma, psi, eps)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**One-sector scale-effects model**")
            st.write("g(L) increases with L.")
        with col2:
            st.markdown("**Variable-variety model**")
            st.write(f"g without scale effects = {g_noscale:.4f} (independent of L).")

        fig1, ax1 = plt.subplots()
        ax1.plot(L_values, g_scale, label="with scale effects")
        ax1.axhline(g_noscale, linestyle="--", label="without scale effects")
        ax1.set_xlabel("population size L")
        ax1.set_ylabel("long-run growth g")
        ax1.set_title("Scale effects vs no-scale effects")
        ax1.legend()
        st.pyplot(fig1)

        L_for_M = st.sidebar.slider("L for M_t path", float(L_min), float(L_max), 2.0, 0.1)
        time_M, M_path = simulate_M_path(L_value=L_for_M, psi=psi, eps=eps, T=100, M0=10.0)

        fig2, ax2 = plt.subplots()
        ax2.plot(time_M, M_path)
        ax2.set_xlabel("time")
        ax2.set_ylabel("M_t")
        ax2.set_title("Number of product varieties M_t over time")
        st.pyplot(fig2)


if __name__ == "__main__":
    main()