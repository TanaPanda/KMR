import numpy as np
import matplotlib.pyplot as plt
import random
from __future__ import division
from mc_tools import mc_compute_stationary, mc_sample_path
from discrete_rv import DiscreteRV

payoff = np.array([[[4, 4], [0, 3]], [[3, 0], [2, 2]]]) # payoff table
N = 10 # the number of players
times = 100000 # the number of repetitions
epsilon = 0.1 # the probability to become mutated
x_0 = 0 # x_t means the number of players playing action1 at time t

payoff_0 = np.array([[payoff[0, 0, 0], payoff[0, 1, 0]], [payoff[1, 0, 0], payoff[1, 1, 0]]]) # payoff matrix for player 0

# setting up transitive matrix A
A = np.zeros([N+1, N+1]) 
A[0, 0] = 1 - epsilon*0.5
A[0, 1] = epsilon*0.5
A[N, N-1] = epsilon*0.5
A[N, N] = 1 - epsilon*0.5
for i in range(1, N): 
    x_i = i / N
    ratio = np.array([1- x_i, x_i]) # the ratio b/w action0 and action1 
    exp = np.dot(payoff_0, ratio)
    if exp[0] > exp[1]:
        A[i, i -1] = x_i * (1-epsilon*0.5)
        A[i, i+1] = (1-x_i) * epsilon*0.5
        A[i, i] = 1- A[i, i-1] - A[i, i+1]
    elif exp[0] < exp[1]:
        A[i, i -1] = x_i * epsilon*0.5
        A[i, i+1] = (1 - x_i) * (1 - epsilon * 0.5)
        A[i, i] = 1- A[i, i-1] - A[i, i+1]
    else:
        A[i, i -1] = x_i * 0.5
        A[i, i+1] = x_i * 0.5
        A[i, i] = 0.5

y = mc_sample_path(A, x_0, times)

fig, ax = plt.subplots()
ax.plot(y)
plt.show()
