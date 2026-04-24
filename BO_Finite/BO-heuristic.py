# %%
import numpy as np
import gymnasium as gym
from gymnasium import spaces
# from stable_baselines3 import PPO, SAC, TD3, DDPG
# from stable_baselines3.common.noise import NormalActionNoise
# from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
print("hi")
# ======================
# Continuous Pricing Environment
# ======================

class ContinuousPricingEnv(gym.Env):
    def _init_(self, min_price=1.0, max_price=100.0, episode_length=100):
        super()._init_()
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(1,),
            dtype=np.float32
        )
        
        self.price_bounds = (min_price, max_price)
        self.episode_length = episode_length
        self.current_step = 0
        self.total_reward = 0.0 # Track cumulative rewards
        self.rewards_history = []
        self.inventory=80

    def reset(self, **kwargs):
        self.current_step = 0
        self.total_reward = 0.0
        self.inventory=80
        self.episode_length = 100

        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        # print(action)
        # min_action = 1
        # max_action = 100
        # price = min_action + (action + 1.0) * 0.5 * (max_action - min_action)
        price = action
        demand = self._demand_function(price)
        reward = price * min(demand,self.inventory)
        self.inventory-= min(demand,self.inventory)
        self.current_step += 1
        self.total_reward += reward
        truncated = False
        terminated = self.current_step >= self.episode_length
        next_state = np.array([demand], dtype=np.float64)
        if terminated:
            self.rewards_history.append(self.total_reward)  # Store reward
        return next_state, reward, terminated, truncated, {"price": price, "demand": demand}

    def _demand_function(self, price):
        lambda_param = -np.log(0.5)/60 #CDF is 0.5 at price 30
        
        customers = int(np.random.poisson(5))
        count = 0
        
        for i in range(customers):
            # Sample willingness to pay from exponential distribution
            willingness_to_pay = np.random.exponential(scale=1/lambda_param)
            
            # Customer buys if price is less than willingness to pay
            if price <= willingness_to_pay:
                count += 1
            
        return count
    
    def get_rewards_history(self):
        return self.rewards_history

# %%
### BO 
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP,HeteroskedasticSingleTaskGP 
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement,UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
import gpytorch


# %%

S= 100
C= 80
def find_new_price(gp,c,s,t):
    # uniform distribution between 1 and 20

    test_X = torch.empty((1000, 1), dtype=torch.float64).uniform_(1, 100)
    # test_X = torch.linspace(1,20,1000).view(-1,1)  
    with torch.no_grad():
        posterior = gp.posterior(test_X)
        mean = posterior.mean
        std = posterior.variance.sqrt()
      
        demand_estimates = mean.view(-1) 

        expected_demand = torch.min((S - s+1) * demand_estimates, torch.tensor(c, dtype=torch.float64))
        necessary_demand = c/ (S - s+1)
        # find all test_X where expected_demand <= necessary_demand
        # print(expected_demand,necessary_demand)
        # test_X[expected_demand <= necessary_demand]
        # revenue = test_X[expected_demand <= necessary_demand] * expected_demand[expected_demand <= necessary_demand]
        # best_index = revenue.argmax().item()
        # best_price = test_X[best_index].item()
        
        expected_revenue = test_X.view(-1) * expected_demand
        
        # plt.plot(test_X.view(-1),expected_revenue)
        # plt.show()
        std = std.view(-1)
        # multiply with test_X
        # std = std * test_X.view(-1)
        acquisition_values = expected_revenue + np.exp(-1*s*t)*std # UCB-based selection
        # elif acquisition == "EI":
        #     best_revenue = expected_revenue.max().item()
        #     EI = ExpectedImprovement(gp, best_f=best_revenue)
        #     acquisition_values = EI(test_X).view(-1)
        # else:
        #     raise ValueError("Acquisition function must be 'UCB' or 'EI'")

        # Select price that maximizes the acquisition function
        best_index = acquisition_values.argmax().item()
        best_price = test_X[best_index].item()
        

    return best_price
# return_array = []

# bucket size 0.1



# bucket size 0.1
env = ContinuousPricingEnv()
from tqdm import tqdm
def bo_episode(C,S,p_array,d_array,p_bucket,d_bucket,t,env):
  c = C
  s = 1
  returns = []
  while s<=S and c>0:
    d = torch.tensor([d_bucket[i][0]/d_bucket[i][1] for i in p_bucket])
    # print(s,c)
    # print(len(p_bucket))
    # print(d.shape)
    gp = SingleTaskGP(torch.tensor(p_bucket).reshape(-1,1),torch.tensor([d_bucket[i][0]/d_bucket[i][1] for i in p_bucket],dtype=torch.float64).reshape(-1,1),outcome_transform=Standardize(m=1),input_transform=Normalize(d=1,bounds=torch.tensor([[1.], [100.]], dtype=torch.float64)))
    # gp = SingleTaskGP(torch.tensor(p_bucket).reshape(-1,1),d.reshape(-1,1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    # season = np.floor(t/S).astype(int)
    p_new = find_new_price(gp,c,s,t)
    
    # demand,_,_,_,_ = env.step(p_new)
    demand = env._demand_function(p_new)
    # print(p_new,prob)
    # print(s,c,p_new,demand)
    if demand!=0:
      d_new = demand
      # returns += p_new*min(d_new,c)
      returns.append(p_new*min(d_new,c))
      s+=1
      c-=min(d_new,c)
    else:
      d_new =  0
      s+=1
      # returns+=0
      returns.append(0) 
    p_array.append(p_new)
    d_array.append(d_new)
    # p_bucket.append(np.round(p_new,1))
    if(np.round(p_new,1) not in d_bucket.keys()):
        p_bucket.append(np.round(p_new,1))
        d_bucket[np.round(p_new,1)] = [d_new,1]
    else:
        # print(p_new,np.round(p_new,0))
        d_bucket[np.round(p_new,1)][0]+=d_new
        d_bucket[np.round(p_new,1)][1]+=1
    # value_matrix,policy_matrix = value_iteration_matrix(beta1,beta2,pl,ph)
    # returns = value_matrix[0,0]
    # plt.plot(graph1)
    # plt.plot(graph2)
  while(s<=S):
      s+=1
      returns.append(0)
    # plt.show()
  # print(t,s,c)

  return returns,p_array,d_array,p_bucket,d_bucket

env = ContinuousPricingEnv()
return_array_final_BO = []
for times in tqdm(range(15)):
  return_array = []
  p1 = np.random.uniform(1,50)
  p_array = []
  p_bucket = []
  p_array.append(p1)
  p2 = np.random.uniform(50,100)
  p_array.append(p2)
  p_bucket.append(np.round(p1,1))
  p_bucket.append(np.round(p2,1))
  d_bucket={}
  d_array = []
  env.reset()
  # choose 1 or 0 based on probability
  d1 = env._demand_function(p1)
  d2 = env._demand_function(p2)
  if(p_bucket[0] not in d_bucket.keys()):
    d_bucket[p_bucket[0]] = [d1,1]
  d_array.append(d1)
  if(p_bucket[1] not in d_bucket.keys()):
    d_bucket[p_bucket[1]] = [d2,1]
  else:
    d_bucket[p_bucket[1]][0]+=d2
    d_bucket[p_bucket[1]][1]+=1
  d_array.append(d2)
  for t in tqdm(range(100)):
    returns,p_array,d_array,p_bucket,d_bucket = bo_episode(C,S,p_array,d_array,p_bucket,d_bucket,t,env)
    return_array.extend(returns)
  return_array_final_BO.append(return_array)
  r = []
  for i in range(0,len(np.mean(return_array_final_BO,axis = 0)),100):
      r.append(np.sum(np.mean(return_array_final_BO,axis = 0)[i:i+100]))

  r = np.array(r)
  r = np.insert(r,0,0)
  import json
  with open(f'bo_after_b_{times}.json', 'w') as f:
      json.dump(r.tolist(), f)
  