import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:15]
#x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


train_err = []
test_err = []




# Data without Normalization
r, c=x.shape
for i in range(0,c ):
    (w, train_error) = a1.linear_regression(x_train[:,i],t_train,'polynomial',0,3)
    (t_est, test_error) = a1.evaluate_regression(x_test[:,i], t_test, w, 'polynomial', 3)
    train_err.append( train_error)
    test_err.append(test_error)


plt.bar(np.arange(8),train_err ,0.2, alpha=0.7, color='b')
plt.bar(np.arange(8)+0.21,test_err ,0.2, color='r', alpha=0.7)
plt.xticks(np.arange(c)+0.1,[('F'+ str(k)) for k in np.arange(8,c+8)])
plt.ylabel('RMS')
plt.legend(['Training error','Test error'])
plt.title('Single featur  \n   polynominal degree = 3  \n[not regularization          not normalized]')
plt.xlabel('Features 8 to 15')
plt.show()    