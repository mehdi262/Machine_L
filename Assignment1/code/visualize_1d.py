#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:, :]
#x = a1.normalize_data(x)

N_TRAIN = 100;
# Select a single feature.
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]



def data_visualization(f):
    x_train = x[0:N_TRAIN, f]
    x_test = x[N_TRAIN:, f]
    # Plot a curve showing learned function.
    # Use linspace to get a set of samples on which to evaluate
    x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
   
    # TO DO:: Put your regression estimate here in place of x_ev.
    # Evaluate regression on the linspace samples.
    (w, tr_err) = a1.linear_regression(x_train, t_train, 'polynomial',0, 3)
    phi = a1.design_matrix(np.transpose(np.asmatrix(x_ev)),'polynomial', 3)
    y_ev = np.transpose(w) * np.transpose(phi)

    plt.plot(x_train, t_train, 'bo',x_test, t_test, 'gs',x_ev, np.transpose(y_ev), 'r.-')

    plt.xlabel(features[f])
    plt.ylabel('Under-5 mortality rate')
    plt.legend([ 'Training data points', 'Test data points','Learned Polynomial'])
    plt.title('A visualization of a regression estimate using random outputs')
    plt.show()
    print('')



data_visualization(10)
data_visualization(11)
data_visualization(12)