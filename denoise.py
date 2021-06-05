#!/usr/local/bin/python

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from scipy.stats import norm

#Load Data
datafile = open('X.pkl', 'rb')
X = pickle.load(datafile)

plt.imsave('output/000.png', X, cmap='cividis')

X = np.pad(X, 1, 'constant')

m, n = np.shape(X)

#Constants
T = 100
alpha = 250
b = 62.5
tau = 0.01

#Index Arrays
even, odd = np.zeros((m,n), dtype=np.bool_), np.zeros((m,n), dtype=np.bool_)
for i in range(1, m-1):
    for j in range(1, n-1):
        if (i+j)%2 == 0:
            even[i,j]=1
        else:
            odd[i,j]=1

Z_prev = np.copy(X)

for t in range(1,T+1):
    print('Iteration:', t)
    Z = np.copy(Z_prev)
    delta = np.reshape(norm.rvs(0, 1/(alpha+tau), size=m*n), (m, n)) 

    S = np.roll(Z,-1,axis=0) + np.roll(Z,1,axis=0) + np.roll(Z,-1,axis=1) + np.roll(Z,1,axis=1)
    Z[even] = ((tau/(alpha+tau))*X[even]) + ((b/(alpha+tau))*S[even]) + delta[even]

    S = np.roll(Z,-1,axis=0) + np.roll(Z,1,axis=0) + np.roll(Z,-1,axis=1) + np.roll(Z,1,axis=1)
    Z[odd] = ((tau/(alpha+tau))*X[odd]) + ((b/(alpha+tau))*S[odd]) + delta[odd]

    plt.imsave('output/'+'{:03d}'.format(t)+'.png', Z[1:-1,1:-1], cmap='cividis')
    Z_prev = np.copy(Z)

