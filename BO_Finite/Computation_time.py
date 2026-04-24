import numpy as np
import torch
import time
from collections import defaultdict
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from scipy.stats import norm
import matplotlib.pyplot as plt
from itertools import product

torch.set_default_dtype(torch.float64)

T = 5  # short run for timing
pl, ph = 1.0, 20.0
bucket_width = 1.0
prices_grid = np.linspace(pl, ph, 100)
beta_true = np.array([2.0, -0.4], dtype=np.float64)

def true_demand(p):
    z = beta_true[0] + beta_true[1] * p
    return 1 / (1 + np.exp(-z))

def update_buckets(p, d, bucket_sum, bucket_count):
    idx = int((p - pl) // bucket_width)
    bucket_sum[idx] += d
    bucket_count[idx] += 1

def get_bucket_data(bucket_sum, bucket_count):
    Xb, yb = [], []
    for idx in bucket_sum:
        Xb.append([pl + (idx + 0.5) * bucket_width])
        yb.append(bucket_sum[idx] / bucket_count[idx])
    return torch.tensor(Xb), torch.tensor(yb).unsqueeze(-1)

def fit_gp_buckets(bucket_sum, bucket_count):
    X, y = get_bucket_data(bucket_sum, bucket_count)
    model = SingleTaskGP(X, y, input_transform=Normalize(1), outcome_transform=Standardize(1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model

def compute_transition_probs(mu, sigma2, c):
    sigma = np.sqrt(sigma2)
    ps = []
    for q in range(c+1):
        lower = (q - 0.5 - mu) / sigma
        upper = (q + 0.5 - mu) / sigma
        if q == 0:
            ps.append(norm.cdf(upper))
        elif q == c:
            ps.append(1 - norm.cdf(lower))
        else:
            ps.append(norm.cdf(upper) - norm.cdf(lower))
    ps = np.array(ps)
    ps /= ps.sum()
    return ps

def value_iteration(gp, C, S):
    V = np.zeros((C+1, S+2))
    policy = np.zeros((C+1, S+1))
    with torch.no_grad():
        X = torch.tensor(prices_grid).unsqueeze(-1)
        mu = gp.posterior(X).mean.squeeze().numpy()
        var = gp.posterior(X).variance.squeeze().numpy()
    for s in range(S, 0, -1):
        for c in range(1, C+1):
            best_val = -np.inf
            for i, p in enumerate(prices_grid):
                trans = compute_transition_probs(mu[i], var[i], c)
                val = sum(trans[q] * (p*q + V[c-q, s+1]) for q in range(len(trans)))
                if val > best_val:
                    best_val = val
                    policy[c, s] = p
            V[c, s] = best_val
    return policy

def heuristic_price(gp, periods_left, c, t, season, S):
    with torch.no_grad():
        X = torch.tensor(prices_grid).unsqueeze(-1)
        mu = gp.posterior(X).mean.squeeze().numpy()
        var = gp.posterior(X).variance.squeeze().numpy()
    exploration = np.exp(-season * t / S) * var
    exp_demand = np.minimum(c, periods_left * mu)
    score = prices_grid * exp_demand + exploration
    return prices_grid[np.argmax(score)]

def run_experiment(algo, C, S):
    bucket_sum = defaultdict(float)
    bucket_count = defaultdict(int)
    for p in [4.0, 7.0]:
        d = np.random.binomial(1, true_demand(p))
        update_buckets(p, d, bucket_sum, bucket_count)
    # print(f"Running {algo} with C={C}, S={S}")
    start = time.time()
    for season in range(T):
        gp = fit_gp_buckets(bucket_sum, bucket_count)
        if algo == "heuristic":
            c = C
            for t in range(1, S + 1):
                if c == 0: break
                p = heuristic_price(gp, S - t + 1, c, t, season, S)
                d = np.random.binomial(1, true_demand(p))
                update_buckets(p, d, bucket_sum, bucket_count)
                c -= min(d, c)
        else:
            policy = value_iteration(gp, C, S)
            c = C
            for t in range(1, S + 1):
                if c == 0: break
                p = policy[c, t]
                d = np.random.binomial(1, true_demand(p))
                update_buckets(p, d, bucket_sum, bucket_count)
                c -= min(d, c)
    # print(f"Finished {algo} with C={C}, S={S}")
    return (time.time() - start) / T

# Configs
C_vals = [5, 10, 20, 40, 80]
S_vals = [10, 20, 40, 80, 160]

results = []

from tqdm import tqdm
# Run experiments and collect results
for C, S in tqdm(product(C_vals, S_vals)):
    if C >= S:
        continue
    t1 = run_experiment("heuristic", C, S)
    t2 = run_experiment("model", C, S)
    results.append((C, S, C*S, t1, t2))

# Convert to structured numpy array
results = np.array(results, dtype=[('C', int), ('S', int), ('CS', int),
                                   ('Heuristic', float), ('ModelBased', float)])
np.savez("computation_comparision.npz", results=results)

# === PLOTTING ===
plt.figure(figsize=(18, 5))

# Fix S for C vs Time (e.g., choose median S)
fixed_S = 30
subset_C = results[results['S'] == fixed_S]

plt.subplot(1, 3, 1)
plt.plot(subset_C['C'], subset_C['Heuristic'], 'o-', label='BO-Fin-Heuristic', linewidth=2)
plt.plot(subset_C['C'], subset_C['ModelBased'], 's--', label='BO-Fin-Model-Based', linewidth=2)
plt.xlabel("C (fixed S={})".format(fixed_S))
plt.ylabel("Time per Season (s)")
plt.title("Runtime vs C (S fixed)")
plt.grid(True)
plt.legend()

# Fix C for S vs Time (e.g., choose median C)
fixed_C = 10
subset_S = results[results['C'] == fixed_C]

plt.subplot(1, 3, 2)
plt.plot(subset_S['S'], subset_S['Heuristic'], 'o-', label='BO-Fin-Heuristic', linewidth=2)
plt.plot(subset_S['S'], subset_S['ModelBased'], 's--', label='BO-Fin-Model-Based', linewidth=2)
plt.xlabel("S (fixed C={})".format(fixed_C))
plt.ylabel("Time per Season (s)")
plt.title("Runtime vs S (C fixed)")
plt.grid(True)
plt.legend()

# CS vs Time using all data
plt.subplot(1, 3, 3)
plt.plot(results['CS'], results['Heuristic'], 'o-', label='BO-Fin-Heuristic', linewidth=2)
plt.plot(results['CS'], results['ModelBased'], 's--', label='BO-Fin-Model-Based', linewidth=2)
plt.xlabel("C × S")
plt.ylabel("Time per Season (s)")
plt.title("Runtime vs C×S")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

