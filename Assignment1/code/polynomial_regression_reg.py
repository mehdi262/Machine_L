import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
(countries, features, values) = a1.load_unicef_data()
targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)
#normalize the input not the target 
N_TRAIN = 100;

x = x[0:N_TRAIN,:]
targets = targets[0:N_TRAIN]

test_arr = dict()

# Split a dataset into k=10 folds
def cross_validation_split(ds_x,ds_target,HOO=1):
    if HOO==1:
        x_train = ds_x[10:100,:]
        t_train = ds_target[10:100]
        x_test = ds_x[0:10:,:]
        t_test = ds_target[0:10]
    elif HOO==10:
        x_train = ds_x[0:90,:]
        t_train = ds_target[0:90]
        x_test = ds_x[90:100,:]
        t_test = ds_target[90:100]
    else:
        x_train = ds_x[0:(HOO-1)*10,:]
        x_train=np.concatenate((x_train, ds_x[ HOO*10 :100,:]), axis=0)
        t_train = ds_target[0:(HOO-1)*10]
        t_train=np.concatenate( (t_train, ds_target[ HOO*10 :100]), axis=0)		
        x_test = ds_x[(HOO-1)*10:HOO*10,:]
        t_test = ds_target[(HOO-1)*10:HOO*10]
    return x_train,t_train,x_test,t_test


def cross_validation(lamb,degree):
    x_train=[]
    t_train=[]
    x_test=[]
    t_test=[]

    test_err = 0
    for i in range(0,10):
        #print('iteration:',i)

        x_train, t_train,x_test,t_test = cross_validation_split(x,targets,i)

        x_train_design = a1.design_matrix(x_train,'polynomial',degree)
        w = np.linalg.inv(lamb * np.identity(x_train_design.shape[1]) + np.transpose(x_train_design)*(x_train_design)) \
            *(np.transpose(x_train_design))*(t_train)

        x_test_design = a1.design_matrix(x_test,'polynomial',degree)
        y_test = np.transpose(w)*np.transpose(x_test_design)
        t_test_error = t_test - np.transpose(y_test)
        rms_test_error = np.sqrt(np.mean(np.square(t_test_error)))

        test_err += rms_test_error
    test_arr[lamb] = test_err/10
    print(test_arr)


for lamb in (0, .01, .1, 1, 10, 100, 1000, 10000):
    cross_validation(lamb,2) 


label = sorted(test_arr.keys())
error = []
for key in label:
    error.append(test_arr[key])

plt.semilogx(label, error)
plt.ylabel('Avg RMS')
plt.legend(['Avg Validation error'])
plt.title('polynomial degree = 2, regularization with 10-folds cross validation')
plt.xlabel('lambda')
plt.show()
