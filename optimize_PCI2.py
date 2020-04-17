from scipy import *
from scipy import optimize
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

#Set the transition probaility for the Perturbed Coin Process
p = 0.75

acc = 1E-15
pmax = 1.0-acc
pmin = 0.0+acc

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
    Statistical complexity of PCI2
    '''
    q1, q2 = x
    q = q1+q2
    return -q1*np.log2(q1/q)/q- q2*np.log2(q2/q)/q

def der_cmu_Q(x):
    '''
    Derivative of cmu_Q
    '''
    q1, q2 = x
    q = q1+q2
    
    df_dq1 = (q1*np.log(q1/q)/q**2 + q2*np.log(q2/q)/q**2 + q2/q**2 + q1/q**2 - 1/q - np.log(q1/q)/q)/np.log(2)
    df_dq2 = (q2*np.log(q2/q)/q**2 + q1*np.log(q1/q)/q**2 + q1/q**2 + q2/q**2 - 1/q - np.log(q2/q)/q)/np.log(2)
    return array([df_dq1, df_dq1])

def cons_c(x):
    '''
    Constraint function. KL divergence rate (closed-form expression for Markov chains)
    '''
    q1, q2 = x
    return [0.5*(p*np.log2(p/q1) + p*np.log2(p/q2) + (1-p)*np.log2((1-p)/(1-q1)) +  (1-p)*np.log2((1-p)/(1-q2)))]

def cons_J(x):
    '''
    Jacobian of the constraint function
    '''
    q1, q2 = x
    return [[(-p/q1 + (1-p)/(1-q1))/(2*np.log(2)), (-p/q2 + (1-p)/(1-q2))/(2*np.log(2))]]

def cons_H(x, v):
    '''
    Hessian matrix of the constraint function=
    '''
    q1, q2 = x   
    return v[0]*array([[(p/q1**2 + (1-p)/(1-q1**2))/(2*np.log(2)), 0], [0, (p/q2**2 + (1-p)/(1-q2**2))/(2*np.log(2))]])

# %% Data acquisition: minimal statistical complexity as a function of delta (maximum distance limit)

N = 50
delta = np.linspace(0.001, 0.75, N)

bounds = optimize.Bounds([pmin, pmin], [pmax, pmax])

q_optimal = np.array([])
cmu_optimal = np.array([])
for i in range(N):
    R = delta[i]*2
    while np.abs(R-delta[i])>1E-5:
        print(R, delta[i])
        x0 = initialize_x(2, init_type='normal')
        
        nonlinear_constraint = optimize.NonlinearConstraint(cons_c, lb=0.0, ub=delta[i], jac='3-point', hess=optimize.BFGS())
        res = optimize.minimize(cmu_Q, x0, method='trust-constr',
                       constraints=[nonlinear_constraint], tol=1E-15,
                       options={'verbose': 1, 'xtol':1E-15, 'gtol':1E-15, 'maxiter':2500},
                       bounds=bounds)
        q = np.asarray(np.real(res.x))
        R = cons_c(q)

    cmu = entropy(q, base=2)
    cmu_optimal = np.append(cmu_optimal, cmu)
    
plt.figure(figsize = [10,6.4])
plt.xlabel('$\\delta$', fontsize=15)
plt.ylabel('$C_{\\mu}(Q^*)$', fontsize=15)
plt.tick_params('both', labelsize=12)

plt.plot(delta, cmu_optimal, 'o', ms=5)

plt.show()

#np.savez('optimal PCI2 RKL variousdelta.npz', d=delta, cmu=cmu_optimal)
# %%