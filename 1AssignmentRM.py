# DYNAMIC PROGRAMMING & REINFORCEMENT LEARNING
# ____________________________________________
# -> ASSIGNMENT 1 = RM problem
# 
# authors: Srishti Nigam, Ander Eguiluz
#
# © Vrije Universiteit 2021
# _________________________

# Problem:
# --------
# Revenue Management of airline company
# Capacity [Seats in the plane 0-100] -> (C) = 100
# X = C-sold tickets -> remaining capacity
# Short time periods [iterations]-> (T) = 600 
# - at most 1 request/period
# Possible prices of tickets f1>f2>f3... -> f = (500, 300, 200) 
# lambda_t(i) -> Probability of request depends on class and time 
# V Function
# - V_600(x) = 0 | departure
# - V_t(0) = 0 | no more seats available
# Target -> calculate Vt(C)

import math
import numpy as np
import matplotlib.pyplot as plt
import random

# VARIABLES
# ---------
# Probability vectors
u = [0.001, 0.015, 0.05]
v = [0.01, 0.005, 0.0025]
# Capacity & Time
C = 100
T = 600
# Fares vector
f = [500, 300 , 200]
# Sumatories
lambda_sum = 0
sec_sum = 0
max_sum = -math.inf
# Matrixes
v_matrix = np.zeros((C+1, T+1))
p_matrix = np.zeros((C+1, T+1))

# task a)
# TOTAL REVENUE
# -------------
# Determine the total expected revenue and the optimal policy
# by implementing dynamic programming yourself in python

def fun_lambda(i, t):
    global u
    global v
    return u[i]*pow(math.e, v[i]*(t + 1))


for t in range(T-1,-1,-1):
    for x in range(C,0,-1):
        for a in range(len(f)):
            for i in range(a+1):
                lambda_sum += fun_lambda(i, t)                
            act_sum = lambda_sum* (f[a] + v_matrix[x-1][t+1]) + (1 - lambda_sum)* v_matrix[x][t+1]
            if act_sum > max_sum:
                max_sum = act_sum
                policy = f[a]
            lambda_sum = 0            
        v_matrix[x][t] = max_sum
        p_matrix[x][t] = policy
        max_sum = -math.inf

# Expected revenue
print( v_matrix[100][1])

# task b)
# PLOTTING 
# --------
# b) Makes a plot of the policy (with time and capacity on the axes)
# TODO uncomment when finishing c,d
# im = plt.imshow(p_matrix, cmap="copper_r")
# #plt.colorbar(im)
# plt.show()

# task c)
# SIMULATION
# ----------
# Simulate the demand over time, and for this realization
# determine which tickets are sold, what the remaining
# capacity is and what the prices are at each moment

S = 10 # n simulations
c = 100 # initial capacity
rewards = []
tot_rewards = []

for s in range(S):
    for t in range(T+1):
        if c != 0:
            # simulated demand:
            # pol = policy[c][t] | lambda at [i][t] = P
            # i = 0,1,2 (500, 300, 200)
            # lambda(i>pol) --> P
            # calcP = random value between 0 and 1
            # calcP > 1 - P  ==> buy the tickt
            lambda_sum = 0
            policy = p_matrix[c][t]
            if policy != 0:
                index = f.index(policy)
                for i in range(index+1):
                    lambda_sum += fun_lambda(i, t)
                act_probability = random.uniform(0,1)                
                # if demand is equal or higher than the policy, seat is purchased 
                if act_probability >= (1 - lambda_sum):
                    c -= 1
                    rewards.append(policy)
                else:
                    rewards.append(0)
            else:
                rewards.append(0)
        else:
            t = 600
    tot_rewards.append(rewards)
    rewards = []
    c = 100

tot = 0
mean = 0

for i in range(len(tot_rewards)):
    sum_rew = 0
    rew_list = tot_rewards[i] 
    for e in range(len(rew_list)):
        sum_rew += rew_list[e]
    tot += sum_rew
#     print(sum_rew)
# mean = tot/len(tot_rewards)


# task d)
# RESTRICTIONS ON POLICY
# ---------------------- 
# Repeat a & c but now for the model where the price cannot
# go down. What is the difference in expected revenue?

v_matrix = np.zeros((C+1, T+1))
p_matrix = np.zeros((C+1, T+1))
 
for t in range(T-1,-1,-1):
    for x in range(C,0,-1):
        for a in range(len(f)):
            for i in range(a+1):
                lambda_sum += fun_lambda(i, t)                
            if x<100 and t<601:
                if f[a] >= p_matrix[x+1][t+1]: #TODO we're close but not there
                    act_sum = lambda_sum* (f[a] + v_matrix[x-1][t+1]) + (1 - lambda_sum)* v_matrix[x][t+1]
            else:
                act_sum = lambda_sum* (f[a] + v_matrix[x-1][t+1]) + (1 - lambda_sum)* v_matrix[x][t+1]
            if act_sum > max_sum:
                max_sum = act_sum
                policy = f[a]
            lambda_sum = 0            
        v_matrix[x][t] = max_sum
        p_matrix[x][t] = policy
        max_sum = -math.inf

# Expected revenue
print( v_matrix[100][1])