
import numpy as np
from scipy import linalg

acc = 1E-10
pmin = 0.0+acc
pmax = 1.0+acc

# %% Useful functions

def stat_dist(T):
    '''
    Compute the stationary distribution of an epsilon-machine
    based on its stochastic matrix
    
    Parameters
    ----------
    T: (n, M, M) array_like
        A 3D array with each transition probabilities arrange along
        the first dimension in the same order as symbols
    
    Returns
    -------
    x: array_like
       The normalized stationary distribution 
    '''
    Ttot = 0
    for i in range(len(T)):
        Ttot += T[i]
    
    w, v = linalg.eig(Ttot, left=True, right=False)
    eig1_index = [i for i, x in enumerate(w) if abs(x-1)<=1E-3]
    stat_dist = v[:, eig1_index[0]]
    return stat_dist / np.sum(stat_dist)

def divrate_markovchain(T1, T2):
    '''
    Calculates the KL divergence rate at the limit L->infinity
    
    Parameters
    ----------
    T1:  (n, M, M) array_like
         Transition matrices under the law P
    T2:  (n, M, M) array_like
         Transition matrices under the law Q
    '''
    sd1 = stat_dist(T1)
    a = np.sum(T1, axis=0)
    b = np.sum(T2, axis=0)
    size = len(a)
    R = 0
    assert size==len(b)
    for i in range(size):
        for j in range(size):
            R += sd1[i] * a[i,j] * log2(a[i,j]/b[i,j])
    return R

def int_to_binary(num, L):
    '''
    Convert an integer value into binary form
    with L-digit counting from the right
    '''
    assert int(num)==num
    numbin = ''.join(reversed( [str((num >> i) & 1) for i in range(L)] ) )
    return numbin

def normalize_mat(A, acc=1E-8):
    '''
    Return a row stochastic matrix (each row sum up to one).
    Anything smaller than acc will be regarded as zero.
    '''
    m, n = shape(A)
    
    for i in range(m):
        if all(A[i]<acc):   #Ignore row filled with all zeros
            continue
        else:
            A[i] = A[i]/sum(A[i])
    
    return A


# %% Popular stationary stochastic process's HMM

def perturbedcoin(p):
    '''
    Symmetric Perturbed Coin Process
    
    Parameter
    ---------
    p: float
       transition probability between states
    '''
    assert p>=pmin and p<=pmax
    T0 = np.array([[1-p, 0], [p, 0]])
    T1 = np.array([[0, p], [0, 1-p]])
    T = np.array([T0, T1])
    return T

def pci2(p1, p2):
    '''
    pci1 is equal to pci2 simply by setting q1 to be whatever p is
    '''
    assert p1 >= pmin and p1 <= pmax
    assert p2 >= pmin and p2 <= pmax
    T0 = np.array([[1-p1, 0], [p2, 0]])
    T1 = np.array([[0, p1], [0, 1-p2]])
    T = np.array([T0, T1])
    return T

def goldenmean(p=0.5):
    '''
    Golden Mean Process. Default is p=0.5
    '''
    assert p>=pmin and p<=pmax
    T0 = np.array([[p, 0], [1, 0]])
    T1 = np.array([[0, 1-p], [0, 0]])
    T = np.array([T0, T1])
    return T

def rgm_k5():
    '''
    Restricted Golden Mean Process with k=5
    '''
    state_s = 6
    T0 = np.zeros([6,6])
    T0[0,0] = 0.5
    T0[1,2] = T0[2,3] = T0[3,4] = T0[4,5] = T0[5,0] = 1
    T1 = np.zeros([6,6])
    T1[0,1] = 0.5
    T = np.array([T0, T1])
    return T

def evenprocess(p=0.5):
    '''
    Even Process. Default is p=0.5
    '''
    assert p>=pmin and p<=pmax
    T0 = np.array([[p, 0], [0, 0]])
    T1 = np.array([[0, 1-p], [1, 0]])
    T = np.array([T0, T1])
    return T

def snsprocess():
    '''
    Nonunifilar form of SNS Process
    '''
    T0 = np.array([[0.5, 0.5], [0, 0.5]])
    T1 = np.array([[0, 0], [0.5, 0]])
    T = np.array([T0, T1])
    return T

def upsetgambler(q):
    '''
    Upset Gambler Process
    '''
    assert q>=pmin and q<=pmax
    T0 = np.array([[1-q, 0], [q, 0]])
    T1 = np.array([[0, q], [1-q, 0]])
    T = np.array([T0, T1])
    return T


# %%