import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------
# GLOBAL DEFAULTS
# ---------------------------------------------------------------------

CHI_DEFAULT = 2.0     # baseline chi for nondrastic case
PSI_DEFAULT = 0.02    # baseline ψ
EPS_DEFAULT = 0.01    # baseline ε


# ---------------------------------------------------------------------
# 1. CORE FUNCTIONS: PROFIT, INNOVATION, GROWTH
# ---------------------------------------------------------------------

def pi_drastic(alpha):
    """Profit parameter π in drastic one-sector model."""
    return (1.0 - alpha) * alpha ** ((1.0 + alpha) / (1.0 - alpha))


def pi_nondrastic(alpha, chi):
    """Profit parameter π(χ) in nondrastic case."""
    return (chi - 1.0) * (alpha / chi) ** (1.0 / (1.0 - alpha))


def equilibrium_n(lambd, sigma, pi_term):
    """Solve φ'(n) * πL = 1 for φ(n) = λ n^σ."""
    exponent = 1.0 / (sigma - 1.0)
    base = 1.0 / (lambd * sigma * pi_term)
    return base ** exponent


def innovation_probability(lambd, sigma, pi_term):
    """Innovation probability μ = φ(n)."""
    n = equilibrium_n(lambd, sigma, pi_term)
    return lambd * n ** sigma


def long_run_growth(lambd, sigma, pi_term, gamma):
    """Long-run growth rate g = μ (γ − 1)."""
    mu = innovation_probability(lambd, sigma, pi_term)
    return mu * (gamma - 1.0)


# ---------------------------------------------------------------------
# 2. ONE-SECTOR SIMULATION
# ---------------------------------------------------------------------

def simulate_one_sector(A0, T, alpha, gamma, lambd, sigma, L, pi_term, seed=0):
    """
    One-sector quality-ladder model.

    A_{t+1} = γ A_t with probability μ,
              A_t   otherwise.

    Y_t, GDP_t and profits are proportional to A_t L.
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
      A_{i,t+1} = γ A_{i,t} with probability μ,
                  A_{i,t}   otherwise.

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
    """M_{t+1} − M_t = ψ L − ε M_t."""
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
    """g in variable-variety model (profit term ∝ ε/ψ)."""
    pi_val = pi_drastic(alpha)
    pi_term = pi_val * (eps / psi)
    return long_run_growth(lambd, sigma, pi_term, gamma)


# ---------------------------------------------------------------------
# 5. STREAMLIT UI
# ---------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Aghion–Howitt Model Simulator", layout="wide")

    st.title("Aghion–Howitt Model Simulator")

    st.sidebar.header("Baseline parameters")

    α = st.sidebar.slider("α (output elasticity)", 0.01, 0.99, 0.30, 0.01)
    γ = st.sidebar.slider("γ (quality step)", 1.01, 1.50, 1.20, 0.01)
    λ = st.sidebar.slider("λ (research productivity)", 0.10, 1.00, 0.50, 0.05)
    σ = st.sidebar.slider("σ (elasticity in φ)", 0.01, 0.99, 0.50, 0.01)
    L = st.sidebar.slider("Population L", 0.5, 5.0, 1.0, 0.1)
    T = st.sidebar.slider("Number of periods T", 50, 400, 200, 10)
    A0 = st.sidebar.number_input("Initial productivity A₀", 0.1, 5.0, 1.0, 0.1)
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

        sim_d = simulate_drastic_one_sector(A0, T, α, γ, λ, σ, L, seed=int(seed))
        π_val = pi_drastic(α)
        π_term = π_val * L
        μ_theory = innovation_probability(λ, σ, π_term)
        g_theory = long_run_growth(λ, σ, π_term, γ)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Theoretical values**")
            st.write(f"π = {π_val:.4f}")
            st.write(f"μ (theoretical) = {μ_theory:.4f}")
            st.write(f"g (theoretical) = {g_theory:.4f}")
        with col2:
            st.markdown("**Simulation**")
            st.write(f"μ (simulation) = {sim_d['mu']:.4f}")
            st.write(f"mean gₜ = {sim_d['g'].mean():.4f}")

        fig1, ax1 = plt.subplots()
        ax1.plot(sim_d["time"], sim_d["A"], label="Aₜ")
        ax1.plot(sim_d["time"], sim_d["Y"], label="Yₜ")
        ax1.plot(sim_d["time"], sim_d["GDP"], label="GDPₜ")
        ax1.set_xlabel("time")
        ax1.set_ylabel("levels")
        ax1.set_title("Drastic one-sector: Aₜ, Yₜ, GDPₜ")
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(sim_d["time"], sim_d["Pi"])
        ax2.set_xlabel("time")
        ax2.set_ylabel("Πₜ")
        ax2.set_title("Monopoly profits Πₜ")
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        ax3.hist(sim_d["g"], bins=30, edgecolor="black")
        ax3.set_xlabel("gₜ")
        ax3.set_ylabel("frequency")
        ax3.set_title("Distribution of realised growth rates gₜ")
        st.pyplot(fig3)

        st.subheader("Sensitivity of g to λ, γ, α")

        lambda_grid = np.linspace(0.2, 0.8, 15)
        gamma_grid = np.linspace(1.05, 1.4, 15)
        alpha_grid = np.linspace(0.1, 0.5, 21)

        g_lambda = [long_run_growth(lmb, σ, π_term, γ) for lmb in lambda_grid]
        g_gamma = [long_run_growth(λ, σ, π_term, gm) for gm in gamma_grid]

        g_alpha = []
        for a in alpha_grid:
            π_a = pi_drastic(a)
            π_term_a = π_a * L
            g_alpha.append(long_run_growth(λ, σ, π_term_a, γ))

        fig4, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].plot(lambda_grid, g_lambda)
        axes[0].set_xlabel("λ")
        axes[0].set_ylabel("g")
        axes[0].set_title("g(λ)")

        axes[1].plot(gamma_grid, g_gamma)
        axes[1].set_xlabel("γ")
        axes[1].set_title("g(γ)")

        axes[2].plot(alpha_grid, g_alpha)
        axes[2].set_xlabel("α")
        axes[2].set_title("g(α)")

        fig4.tight_layout()
        st.pyplot(fig4)

    # -------------------------------------------------------------
    # One-sector: nondrastic innovations
    # -------------------------------------------------------------
    elif mode == "One-sector: nondrastic innovations":
        st.header("One-sector model with nondrastic innovations")

        χ = st.sidebar.slider("χ (limit price)", 1.01, 10.0, CHI_DEFAULT, 0.05)

        # simulation for chosen χ
        sim_nd = simulate_nondrastic_one_sector(
            A0, T, α, γ, λ, σ, L, χ, seed=int(seed)
        )
        # drastic benchmark
        sim_d = simulate_drastic_one_sector(
            A0, T, α, γ, λ, σ, L, seed=int(seed)
        )

        χ_grid = np.linspace(1.01, 10.0, 200)
        π_χ = []
        g_χ = []
        μ_χ = []

        π_d = pi_drastic(α)
        χ_threshold = 1.0 / α  # boundary χ = 1/α

        for χ_val in χ_grid:
            if χ_val < χ_threshold:
                π_val_χ = pi_nondrastic(α, χ_val)
            else:
                π_val_χ = π_d

            π_term_χ = π_val_χ * L
            μ_val = innovation_probability(λ, σ, π_term_χ)
            g_val = μ_val * (γ - 1.0)

            π_χ.append(π_val_χ)
            μ_χ.append(μ_val)
            g_χ.append(g_val)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Baseline nondrastic case**")
            st.write(f"χ = {χ:.3f}")
            st.write(f"μ (simulation) = {sim_nd['mu']:.4f}")
            st.write(f"mean gₜ = {sim_nd['g'].mean():.4f}")
        with col2:
            st.markdown("**Drastic benchmark**")
            π_term_d = π_d * L
            μ_d = innovation_probability(λ, σ, π_term_d)
            g_d = long_run_growth(λ, σ, π_term_d, γ)
            st.write(f"χ threshold = 1/α ≈ {χ_threshold:.2f}")
            st.write(f"μ (drastic) = {μ_d:.4f}")
            st.write(f"g (drastic) = {g_d:.4f}")

        fig1, axes = plt.subplots(3, 1, figsize=(7, 8))
        axes[0].plot(χ_grid, π_χ)
        axes[0].axvline(χ_threshold, linestyle="--", color="gray")
        axes[0].set_xlabel("χ")
        axes[0].set_ylabel("π(χ)")
        axes[0].set_title("Profit parameter π(χ) (dashed: χ = 1/α)")

        axes[1].plot(χ_grid, μ_χ)
        axes[1].axvline(χ_threshold, linestyle="--", color="gray")
        axes[1].set_xlabel("χ")
        axes[1].set_ylabel("μ(χ)")
        axes[1].set_title("Innovation probability μ(χ)")

        axes[2].plot(χ_grid, g_χ)
        axes[2].axvline(χ_threshold, linestyle="--", color="gray")
        axes[2].set_xlabel("χ")
        axes[2].set_ylabel("g(χ)")
        axes[2].set_title("Long-run growth g(χ)")

        fig1.tight_layout()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(sim_d["time"], sim_d["GDP"], linestyle="--", label="drastic GDPₜ")
        ax2.plot(sim_nd["time"], sim_nd["GDP"], label="nondrastic GDPₜ")
        ax2.set_xlabel("time")
        ax2.set_ylabel("GDPₜ")
        ax2.set_title("GDP: drastic vs nondrastic")
        ax2.legend()
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        ax3.plot(sim_nd["time"], sim_nd["Pi"], label="nondrastic Πₜ")
        ax3.plot(sim_d["time"], sim_d["Pi"], linestyle="--", label="drastic Πₜ")
        ax3.set_xlabel("time")
        ax3.set_ylabel("Πₜ")
        ax3.set_title("Profits: drastic vs nondrastic")
        ax3.legend()
        st.pyplot(fig3)

    # -------------------------------------------------------------
    # Multisector model
    # -------------------------------------------------------------
    elif mode == "Multisector model":
        st.header("Multisector model and smoothing of growth")

        n_sectors = st.sidebar.slider("Number of sectors", 10, 500, 200, 10)

        sim_d = simulate_drastic_one_sector(A0, T, α, γ, λ, σ, L, seed=int(seed))
        sim_m = simulate_multisector(
            A0, T, α, γ, λ, σ, L,
            n_sectors=n_sectors, seed=int(seed)
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Innovation probability**")
            st.write(f"μ (common) = {sim_m['mu']:.4f}")
        with col2:
            st.markdown("**Average growth rates**")
            st.write(f"mean gₜ (one-sector) = {sim_d['g'].mean():.4f}")
            st.write(f"mean gₜ (multisector) = {sim_m['g_agg'].mean():.4f}")

        fig1, ax1 = plt.subplots()
        ax1.plot(sim_m["time"], sim_m["A_agg"], label="Aₜ aggregate")
        ax1.plot(sim_m["time"], sim_m["Y_agg"], label="Yₜ aggregate")
        ax1.plot(sim_m["time"], sim_m["GDP_agg"], label="GDPₜ aggregate")
        ax1.set_xlabel("time")
        ax1.set_ylabel("levels")
        ax1.set_title("Multisector: aggregate variables")
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        for i in range(min(5, n_sectors)):
            ax2.plot(sim_m["time"], sim_m["A_sectors"][:, i])
        ax2.set_xlabel("time")
        ax2.set_ylabel("Aᵢₜ")
        ax2.set_title("Selected sectoral productivity paths")
        st.pyplot(fig2)

        Tg = min(len(sim_d["g"]), len(sim_m["g_agg"]))
        time_g = np.arange(Tg)

        fig3, ax3 = plt.subplots()
        ax3.plot(time_g, sim_d["g"][:Tg], label="one-sector gₜ")
        ax3.plot(time_g, sim_m["g_agg"][:Tg], linestyle="--", label="multisector gₜ")
        ax3.set_xlabel("time")
        ax3.set_ylabel("gₜ")
        ax3.set_title("Growth rates: one-sector vs multisector")
        ax3.legend()
        st.pyplot(fig3)

    # -------------------------------------------------------------
    # Scale effects
    # -------------------------------------------------------------
    elif mode == "Scale effects":
        st.header("Scale effects and population size")

        ψ = st.sidebar.slider("ψ (discovery prob per person)", 0.01, 0.99, PSI_DEFAULT, 0.01)
        ε = st.sidebar.slider("ε (disappearance fraction)", 0.01, 0.99, EPS_DEFAULT, 0.01)

        L_min = st.sidebar.number_input("L grid min", 0.2, 5.0, 0.5, 0.1)
        L_max = st.sidebar.number_input("L grid max", 0.5, 20.0, 5.0, 0.5)
        grid_points = st.sidebar.slider("Grid points for L", 5, 50, 20, 1)
        L_values = np.linspace(L_min, L_max, grid_points)

        g_scale = growth_with_scale_effects(L_values, α, γ, λ, σ)
        g_noscale = growth_without_scale_effects(α, γ, λ, σ, ψ, ε)

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

        L_for_M = st.sidebar.slider("L for Mₜ path", float(L_min), float(L_max), 2.0, 0.1)
        time_M, M_path = simulate_M_path(L_value=L_for_M, psi=ψ, eps=ε, T=100, M0=10.0)

        fig2, ax2 = plt.subplots()
        ax2.plot(time_M, M_path)
        ax2.set_xlabel("time")
        ax2.set_ylabel("Mₜ")
        ax2.set_title("Number of product varieties Mₜ over time")
        st.pyplot(fig2)


if __name__ == "__main__":
    main()