#This script generates the word probability distribution of certain process
#specified by the inputed symbol-labeled transition matrices and save the array
#into csv file

import numpy as np
from scipy.stats import entropy
import hmm
from complexity_function import *

T = hmm.evenprocess()
sd = stat_dist(T)
state_size = len(sd)

def prob_generator(L):
    symbols = ['0', '1']
    perm = ["".join(seq) for seq in itertools.product(symbols, repeat=L)]
    prob_dist = np.zeros(len(perm))
    count = 0
    for w in perm:
        word_mat = np.eye(state_size)
        for i in w:
            emit_symbol = int(i)
            word_mat = np.matmul(T[emit_symbol], word_mat)
        prob_dist[count] = np.sum(np.matmul(sd, word_mat))
        count += 1
    return prob_dist

M = 10
N = 8
L = M+N

p_x = prob_generator(M)
p_y = prob_generator(N)
p_xy = prob_generator(L)

#Finite past-future excess entropy (mutual information)
print('E(M,N): ', entropy(p_xy, np.kron(p_x, p_y), base=2))

#np.savez('snsprocess K10L8.npz', p_x=p_x, p_y=p_y, p_xy=p_xy)
