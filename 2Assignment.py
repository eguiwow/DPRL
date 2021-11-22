# DYNAMIC PROGRAMMING & REINFORCEMENT LEARNING
# ____________________________________________
# -> ASSIGNMENT 2 = RM problem
# 
# author: Ander Eguiluz
#
# © Vrije Universiteit 2021
# _________________________

# Problem:
# --------
# System that slowly deteriorates
# - when new, failure probability = 0.1
# - every time unit -> delta_failure increases = 0.01
# - replacement cost = 1
# - after replacement the part is new
# SOL:
# compute the stationary distribution and use it to find the long-run average replacement costs
# solve the average-cost Poisson equation
# Preventive replacement is possible at cost 0.5
# - what is the avg optimal policy?
# solve this using: policy iteration & value iteration

import math
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
np.set_printoptions(threshold=sys.maxsize)

# VARIABLES
# ---------

# task a)
# STATIONARY DISTRIBUTION, FIND LONG-RUN AVERAGE REPLACEMENT COSTS
# -------------
probability_matrix = np.zeros((91, 91))
i = 0.1
while i<1:
    for x in range(91):
        probability_matrix[0][x] = i
        i+=0.01

for x in range(1,90):
    probability_matrix[x][x-1] = 1-probability_matrix[0][x-1]
    
# print(probability_matrix[0][0])
# print(probability_matrix[1][0])
# print(probability_matrix)
# pi_star = []
# pi_star = [0 for i in range(91)]
# print(pi_star)

# pi_star[0] = 1

# for y in range(91):
#     for x in range(91):
#         pi_star


pi_0 = np.zeros(91)
pi_0[0] = 1 
r = pi_0

# pi_1 = np.zeros(91)
T = 10000

# pi_1 = probability_matrix.dot(pi_0)
# pi_2 = probability_matrix.dot(pi_1)

for t in range(T):
    pi_1 = probability_matrix.dot(pi_0)
    pi_0 = pi_1

#print(pi_1)

# task b)
# SOLVE THE AVG-COST POISSON EQUATION
# -------------

# v_star =  pi_0 = np.zeros(91)

# for i in range(1,90):
#     v_star[i] = (v_star[i-1] + pi_1[0]) / probability_matrix[i+1][i]

# print(v_star)

V = np.zeros(91)

for i in range(100):
    V = r + probability_matrix.dot(V)

value_matrix = V-V.min()
# print(value_matrix)

phi = r +  probability_matrix.dot(V) - V
# print(phi)


# task c)
# PREVENTIVE REPLACEMENT AT COST 0.5
# --------------
# what is the optimal policy?
# using:
# - policy iteration
# - value iteration



V = np.zeros(91)

r1 = np.zeros(91)
r1[0] = 1
r2 = np.zeros(91)
r2[0] = 0.5

diag = np.zeros((91,91))
np.fill_diagonal(diag, 1)

P1 = 0.5 *(probability_matrix + diag)
P2 = 0.5 *(probability_matrix + diag)

for i in range(100):
    V = np.maximum(r1+P1.dot(V), r2+P2.dot(V))


val_matrix = V-V.min()

# change from here!!!
a = np.argmax([r1[13] + P1[13].dot(V), r2[13] + P2[13].dot(V)])
print(a)

if(a==0):
    print(r1 + P1.dot(V) - V)
else:
    print(r2 + P2.dot(V) -V)



