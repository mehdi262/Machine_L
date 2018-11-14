#!/usr/bin/env python

# Run logistic regression training using  different etas.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2


# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
etas = [0.5, 0.3, 0.1, 0.05, 0.01]


# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
X = data[:,0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:,3]
# For plotting data
class1 = np.where(t==0)
X1 = X[class1]
class2 = np.where(t==1)
X2 = X[class2]


plt.figure()

for eta in etas:
    # Initialize w.
    w = np.array([0.1, 0, 0])

    # Error values over all iterations.
    e_all = []

    for iter in range (0,max_iter):
        # Compute output using current w on all data X.
        y = sps.expit(np.dot(X,w))
        # e is the error, negative log-likelihood (Eqn 4.90)
        e = -np.mean(np.multiply(t,np.log(y)) + np.multiply((1-t),np.log(1-y)))

        # Add this error to the end of error vector.
        e_all.append(e)

        # Gradient of the error, using Eqn 4.91
        grad_e = np.mean(np.multiply((y - t), X.T), axis=1)

        # Update w, *subtracting* a step in the error derivative since we're minimizing
        w_old = w
        w = w - eta*grad_e
 
        # Plot current separator and data.  Useful for interactive mode / debugging.
        # plt.figure(DATA_FIG)
        # plt.clf()
        # plt.plot(X1[:,0],X1[:,1],'b.')
        # plt.plot(X2[:,0],X2[:,1],'g.')
        # a2.draw_sep(w)
        # plt.axis([-5, 15, -10, 10])
  
        # Stop iterating if error doesn't change more than tol.
        if iter>0:
            if np.absolute(e-e_all[iter-1]) < tol:
                break
        #print some info
        if (iter % 10==0):
            print ('eta {0:.4f}, inter {1:d}, negative log-likelihood {2:.4f}, w={3}'.format(eta,iter, e, w.T))
    print ('*************************************************************************************************\n')
    plt.plot(e_all)


plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression (Gradient Descent)')
plt.xlabel('Epoch')
legend=[str(eta) for eta in etas]
plt.legend(legend,loc='upper right',  ncol=5)
plt.show()
 