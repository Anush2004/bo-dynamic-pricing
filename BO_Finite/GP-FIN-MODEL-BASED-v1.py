import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from tqdm import tqdm

# Set default precision to float64
torch.set_default_dtype(torch.float64)

# Parameters
C = 10
S = 20
T = 100
M = 50000
pl, ph = 0.1, 20.0
eps = (ph - pl) / 10
prices_grid = np.linspace(pl, ph, 200, dtype=np.float64)
beta_true = np.array([2.0, -0.4,1.0], dtype=np.float64)

# True demand model
def true_demand_prob(p,beta,t):
    # z = beta_true[0] + beta_true[1] * p
    # return 1 / (1 + np.exp(-z))
    z = beta[0] + beta[1]*p + beta[2]*np.log(p/(ph-p+1))

    return 1/(1 + np.exp(-z))

# GP fitting with float64 and proper shapes
def fit_gp(X, y):
    train_X = torch.tensor(X, dtype=torch.float64)                    # (N, 1)
    train_Y = torch.tensor(y, dtype=torch.float64).unsqueeze(-1)  # (N, 1)

    gp = SingleTaskGP(
        train_X, train_Y,
        input_transform=Normalize(1),
        outcome_transform=Standardize(1)
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    return gp

# Value iteration using GP mean - one step - closer to heuristic
def value_iteration_gp(gp):
    V = np.zeros((C + 1, S + 2))
    policy = np.zeros((C + 1, S + 1))

    price_tensor = torch.tensor(prices_grid, dtype=torch.float64).unsqueeze(-1)
    with torch.no_grad():
        mu = gp.posterior(price_tensor).mean.squeeze().numpy()

    for s in range(S, 0, -1):
        for c in range(1, C + 1):
            best_val, best_p = -np.inf, None
            for i, p in enumerate(prices_grid):
                d = mu[i]
                reward = p * d + d * V[c - 1, s + 1] + (1 - d) * V[c, s + 1]
                if reward > best_val:
                    best_val = reward
                    best_p = p
            V[c, s] = best_val
            policy[c, s] = best_p
    return policy, V

# ε-check
def within_eps(prices, target, eps):
    return all(abs(p - target) < eps for p in prices)

# Initialization
X_data, y_data = [], []
regrets, cum_rewards, gp_error_timestep = [], [0.], []
policies = []

# Seed with 2 diverse price points
init_prices = [5.0, 15.0]
for p in init_prices:
    d = np.random.binomial(1, true_demand_prob(p, beta_true, 1))
    X_data.append([p])
    y_data.append(float(d))

# Compute true V*
V_opt = np.zeros((C + 1, S + 2))
for s in range(S, 0, -1):
    for c in range(1, C + 1):
        best_val = -np.inf
        for p in prices_grid:
            d = true_demand_prob(p, beta_true, s)
            val = p * d + d * V_opt[c - 1, s + 1] + (1 - d) * V_opt[c, s + 1]
            best_val = max(best_val, val)
        V_opt[c, s] = best_val
V_star = V_opt[C, 1]

# Main loop
for t in tqdm(range(T)):
    gp = fit_gp(X_data, y_data)
    policy, _ = value_iteration_gp(gp)

    prices_this_season = []
    c = C
    for s in range(1, S + 1):
        if c == 0:
            break

        p_ceq = policy[c, s]
        same_prices = [p for (p, ss) in prices_this_season if ss == s]
        cond_a = all(abs(p1 - p2) < eps for p1 in same_prices for p2 in same_prices) if len(same_prices) >= 2 else True
        cond_b = within_eps(same_prices, p_ceq, eps)
        cond_c = c == 1 or s == S

        if cond_a and cond_b and cond_c:
            candidates = [p_ceq + 2 * eps, p_ceq - 2 * eps]
            candidates = [p for p in candidates if pl <= p <= ph]
            p = np.random.choice(candidates)
        else:
            p = p_ceq

        d = np.random.binomial(1, true_demand_prob(p, beta_true, s))
        X_data.append([p])
        y_data.append(float(d))
        prices_this_season.append((p, s))
        c = max(c - d, 0)

        # Error per timestep
        with torch.no_grad():
            mu_pred = gp.posterior(torch.tensor([[p]], dtype=torch.float64)).mean.item()
        gp_error_timestep.append(abs(mu_pred - true_demand_prob(p, beta_true, s)))

    # Monte Carlo estimate of expected revenue
    total_reward = 0
    for _ in range(M):
        c_sim = C
        reward = 0
        for s in range(1, S + 1):
            if c_sim == 0:
                break
            p = policy[c_sim, s]
            d = np.random.binomial(1, true_demand_prob(p, beta_true, s))
            fulfilled = min(d, c_sim)
            reward += p * fulfilled
            c_sim -= fulfilled
        total_reward += reward
    avg_reward = total_reward / M
    cum_rewards.append(avg_reward)
    regrets.append(V_star - avg_reward)
    policies.append(policy)


# Plotting
plt.figure(figsize=(18, 4))

plt.subplot(1, 3, 1)
plt.plot(gp_error_timestep, label="|μ(p) − D(p)|")
plt.title("GP Estimation Error per Timestep")
plt.xlabel("Timestep")
plt.ylabel("Error")
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(regrets, label="Expected Regret")
plt.title("Expected Regret per Season")
plt.xlabel("Season")
plt.ylabel("Regret")
plt.grid(True)
plt.legend()

cum_regret = np.cumsum(regrets)
# add 0 at the beginning for cumulative regret
cum_regret = np.insert(cum_regret, 0, 0)
plt.subplot(1, 3, 3)
plt.plot(cum_regret, label="Cumulative Regret")
plt.title("Cumulative Regret")
plt.xlabel("Season")
plt.ylabel("Cumulative Regret")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
