import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt 
(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
N_x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

train_err = []
test_err = []


N_x_train = N_x[0:N_TRAIN,:]
N_x_test = N_x[N_TRAIN:,:]
N_train_err = []
N_test_err = []




d=7 # degree= d-1
for i in range(1, d):
    (w, tr_err) = a1.linear_regression(x_train,t_train,'polynomial',0,i)
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, 'polynomial', i)
    train_err.append( tr_err)
    test_err.append( te_err)

for i in range(1, d):
    (w, tr_err) = a1.linear_regression(N_x_train,t_train,'polynomial',0,i)
    (t_est, te_err) = a1.evaluate_regression(N_x_test, t_test, w, 'polynomial', i)
    N_train_err.append(tr_err)
    N_test_err.append(te_err)

# Produce a plot of results.
plt.plot(np.arange(len(train_err)),train_err,color='b')
plt.plot(np.arange(len(test_err)),test_err,color='r')

plt.ylabel('RMS')
plt.legend(['Training error','Test error'])
plt.title('Fit with polynomials,Not normalization')
plt.xlabel('Polynomial degree')
plt.show()

plt.plot(np.arange(len(N_train_err)),N_train_err,color='b')
plt.plot(np.arange(len(N_test_err)),N_test_err,color='r')
plt.ylabel('RMS')
plt.legend(['Training error','Test error'])
plt.title('Fit with polynomials, Normalization')
plt.xlabel('Polynomial degree')
plt.show()