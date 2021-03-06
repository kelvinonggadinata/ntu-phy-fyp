#Optimization for PCI1 with KL divergence rate as the distance measure

import numpy as np
from scipy import optimize
from scipy import random
from scipy.stats import entropy
import matplotlib.pyplot as plt

#Set the transition probability for Perturbed Coin Process
p = 0.75

acc = 1E-15
pmin = 0.0+acc
pmax = 1.0-acc

# %% Functions for the optimization

def initialize_x(N, init_type='normal'):
    '''
    Initialized an array of length N filled with probability parameter.
    
    init_type: str
               Accepts 'equal', 'normal', and 'uniform'
    '''
    x = np.zeros(N)
    initialize = True
    while initialize:
        if init_type=='normal':
            x = random.normal(p, min(p, (1-p))/2, N)
        elif init_type=='uniform':
            x = random.uniform(pmin, pmax, N)
        elif init_type=='equal':
            x = np.full(N, p)
        else:
            print('init_type is wrong')
            exit()
        
        if np.all(x<=pmax) and np.all(x>=pmin):
            initialize = False
        else:
            initialize = True
    
    return x

def cmu_Q(x):
    '''
    Statistical complexity expression for process Q
    
    x: array_like
       Free parameters of process Q. In this case only one element (free parameter)
    '''
    q = x[0]
    return -p*np.log2(p/(p+q))/(p+q) - q*np.log2(q/(p+q))/(p+q)

def cons_c(x):
    '''
    Constraint function: KL divergence rate (closed-form expression for Markov chains)
    '''
    q = x[0]
    return [0.5*(p*np.log2(p/q) + (1-p)*np.log2((1-p)/(1-q)) )]

def cons_J(x):
    '''
    Jacobian of the constraint function
    '''
    q = x[0]
    return [[(-p/q + (1-p)/(1-q))/(2*np.log(2))]]

def cons_H(x, v):
    '''
    Hessian matrix of the constraint function. 
    It is written in a linear combination fashion: H(x,v) = \sum_{i}v_{i}\nabla^2c_{i}(x), 
    where v_{i} is Lagrange multiplier for constraint c_{i}
    
    v: array_like
       Lagrange multipliers
    '''
    q = x[0]   
    return v[0]*array([[(p/q**2 + (1-p)/(1-q**2))/(2*np.log(2))]])

#Bounds for the free parameters
bounds = optimize.Bounds([pmin], [pmax])

# %% Data acquisition: minimal statistical complexity as a function of delta (maximum distance limit)

N = 50
delta = np.linspace(0.001, 0.75, N)

#Create empty array for solutions
q_optimal = np.array([])
cmu_optimal = np.array([])

#Find solutions for every delta
for i in range(N):
    R = delta[i]*2
    while np.abs(R-delta[i])>1E-4:
        print(R, delta[i])
        x0 = initialize_x(1, init_type='normal') #Must be length of one
        
        nonlinear_constraint = optimize.NonlinearConstraint(cons_c, lb=0.0, ub=delta[i], jac='3-point', hess=optimize.BFGS())
        res = optimize.minimize(cmu_Q, x0, method='trust-constr',
                       constraints=[nonlinear_constraint], tol=1E-12,
                       options={'verbose': 1, 'xtol':1E-12, 'gtol':1E-12, 'maxiter':2500},
                       bounds=bounds)
        q = np.asarray(np.real(res.x))
        R = cons_c(q)

    q_optimal = np.append(q_optimal, q)
    cmu = entropy([p, q.item(0)], base=2)
    cmu_optimal = np.append(cmu_optimal, cmu)

#Plot the solutions
plt.figure(figsize = [10,6.4])
plt.xlabel('$\\delta$', fontsize=15)
plt.ylabel('$C_{\\mu}(Q^*)$', fontsize=15)
plt.tick_params('both', labelsize=12)

plt.plot(delta, cmu_optimal, 'o', ms=5)

plt.show()

#np.savez('optimal PCI1 RKL variousdelta.npz', d=delta, cmu=cmu_optimal)
# %% 