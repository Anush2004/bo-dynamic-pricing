import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from tqdm import tqdm
from collections import defaultdict

# --- Setup ---
torch.set_default_dtype(torch.float64)

C, S, T = 10, 20, 100            # inventory, season length, #seasons
M = 50000                        # Monte Carlo sims per season
pl, ph = 1.0, 20.0
bucket_width = 1.0               # price‐bucket width
prices_grid = np.linspace(pl, ph, 200, dtype=np.float64)
beta_true = np.array([2.0, -0.4], dtype=np.float64)

def true_demand(p):
    z = beta_true[0] + beta_true[1]*p
    return 1/(1 + np.exp(-z))

# --- Incremental Bucketing Structures ---
bucket_sum   = defaultdict(float)
bucket_count = defaultdict(int)

def update_buckets(p, d):
    idx = int((p - pl)//bucket_width)
    bucket_sum[idx]   += d
    bucket_count[idx] += 1

def get_bucket_data():
    Xb, yb = [], []
    for idx, total in bucket_sum.items():
        ctr = pl + (idx + 0.5)*bucket_width
        Xb.append([ctr])
        yb.append(total / bucket_count[idx])
    return Xb, yb

# --- GP Fit on Buckets ---
def fit_gp_buckets():
    Xb, yb = get_bucket_data()
    train_X = torch.tensor(Xb, dtype=torch.float64)
    train_Y = torch.tensor(yb, dtype=torch.float64).unsqueeze(-1)
    gp = SingleTaskGP(train_X, train_Y,
        input_transform=Normalize(1),
        outcome_transform=Standardize(1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    return gp

# --- Transition Integration (demand approximation) ---
from scipy.stats import norm
def compute_transition_probs(mu, sigma2, c):
    sigma = np.sqrt(sigma2)
    ps = []
    for q in range(c+1):
        lower = (q-0.5 - mu)/sigma
        upper = (q+0.5 - mu)/sigma
        if q == 0:
            p_q = norm.cdf(upper)
        elif q == c:
            p_q = 1 - norm.cdf(lower)
        else:
            p_q = norm.cdf(upper) - norm.cdf(lower)
        ps.append(p_q)
    ps = np.array(ps); ps /= ps.sum()
    return ps

# --- GP‑based Value Iteration ---
def value_iteration_gp(gp):
    V = np.zeros((C+1, S+2))
    policy = np.zeros((C+1, S+1))
    Xtest = torch.tensor(prices_grid, dtype=torch.float64).unsqueeze(-1)
    with torch.no_grad():
        post = gp.posterior(Xtest)
        mu  = post.mean.squeeze().numpy()
        var = post.variance.squeeze().numpy()
    for s in range(S, 0, -1):
        for c in range(1, C+1):
            best_val, best_p = -np.inf, None
            for i,p in enumerate(prices_grid):
                trans = compute_transition_probs(mu[i], var[i], c)
                exp_rev = 0.0
                for q,prob in enumerate(trans):
                    s_prime = c - q
                    reward = p*q
                    exp_rev += prob*(reward + V[s_prime, s+1])
                if exp_rev > best_val:
                    best_val, best_p = exp_rev, p
            V[c,s], policy[c,s] = best_val, best_p
    return policy, V

# --- Initialization & True Optimum ---
# seed two distinct prices
for p in [4.0, 7.0]:
    d = np.random.binomial(1, true_demand(p))
    update_buckets(p, d)

# compute true optimal seasonal revenue V_star
Vopt = np.zeros((C+1, S+2))
for s in range(S,0,-1):
    for c in range(1, C+1):
        Vopt[c,s] = max(
            p*true_demand(p) 
            + true_demand(p)*Vopt[c-1, s+1]
            + (1-true_demand(p))*Vopt[c, s+1]
            for p in prices_grid
        )
V_star = Vopt[C,1]

# --- Simulation ---
regrets = []; errors = [];policies = []

for season in tqdm(range(T)):
    # fit GP at season start
    gp = fit_gp_buckets()
    policy, _ = value_iteration_gp(gp)

    c, season_rev = C, 0.0
    for t in range(1, S+1):
        if c==0: break
        p = policy[c,t]
        d = np.random.binomial(1, true_demand(p))
        sold = min(d,c)
        season_rev += p*sold
        c -= sold
        update_buckets(p, d)
        # track GP error
        with torch.no_grad():
            mu_pred = gp.posterior(torch.tensor([[p]],dtype=torch.float64)).mean.item()
        errors.append(abs(mu_pred - true_demand(p)))

    # Monte Carlo expected revenue
    total=0.0
    for _ in range(M):
        c_sim, rev = C, 0.0
        for t in range(1,S+1):
            if c_sim==0: break
            p_mc = policy[c_sim,t]
            d_mc = np.random.binomial(1, true_demand(p_mc))
            sold = min(d_mc,c_sim)
            rev += p_mc*sold
            c_sim -= sold
        total+=rev
    exp_rev = total/M
    regrets.append(V_star - exp_rev)
    policies.append(np.array([policy[c,s] for c in range(C+1) for s in range(S+1)]))
# --- Plots ---
plt.figure(figsize=(18,4))
plt.subplot(131)
plt.plot(errors); plt.title("GP Error")
plt.subplot(132)
plt.plot(regrets); plt.title("Regret per Season")
plt.subplot(133)
plt.plot(np.cumsum(regrets)); plt.title("Cumulative Regret")
plt.tight_layout(); plt.show()
