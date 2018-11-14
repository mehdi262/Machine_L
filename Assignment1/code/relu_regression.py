#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
targets = values[:, 1]
x = values[:, 10]

N_TRAIN = 100;

x_train = x[0:N_TRAIN, :]
x_test = x[N_TRAIN:, :]

t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

(w, tr_error) = a1.linear_regression(x_train, t_train, 'ReLU', 0, 1)
phi = a1.design_matrix(x_test, 'ReLU', 1)
y = np.transpose(w) * np.transpose(phi)
error_test = t_test - np.transpose(y)
tes_error = np.sqrt(np.mean(np.square(error_test)))

x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
x_ev = np.transpose(np.asmatrix(x_ev))
phi_ev = a1.design_matrix(x_ev, 'ReLU', 1)
y_ev = np.transpose(w) * np.transpose(phi_ev)

plt.plot(x_train, t_train, 'bo',x_test, t_test, 'gs',x_ev, np.transpose(y_ev), 'r.-')

plt.legend(['Training data', 'Test data', 'Learned Function'])
titl='ReLU Regression \n'
titl= titl +' Train Error :'+str(tr_error)+' \n'
titl=titl+'  Test Error:'+ str(tes_error)
print(titl)
plt.title(titl)

plt.show()



