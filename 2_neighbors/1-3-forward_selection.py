import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import time

# load data
print("Loading training data ...")
data_train = np.genfromtxt("data/train.csv", comments="#", delimiter=",")
data_val = np.genfromtxt("data/validation.csv", comments="#", delimiter=",")


def get_mse(Xtrain, ytrain, Xtrue, ytrue):
    model = KNeighborsRegressor(n_neighbors=10, algorithm="kd_tree")
    model.fit(Xtrain, ytrain)
    preds = model.predict(Xtrue)
    mse = mean_squared_error(y_true=ytrue, y_pred=preds)
    return mse


Xtrain, ytrain = data_train[:,:-1], data_train[:,-1]
Xval, yval = data_val[:,:-1], data_val[:,-1]

n_features = 1
curr_err = 0
i = 0
errs = []
selected = []
# first run 
while i < Xtrain.shape[1]:
    train_x = Xtrain[:,i].reshape(-1, 1)
    test_x = Xval[:,i].reshape(-1, 1)
    errs.append(get_mse(train_x, ytrain, test_x, yval))
    i += 1

first_feature = np.argmin(errs)
curr_err = errs[first_feature]
best_train = Xtrain[:,first_feature].reshape(-1, 1)
best_test = Xval[:,first_feature].reshape(-1, 1)
selected.append(first_feature)

print("N_features: %s, last best feature: %s, current shape: %s, current error: %s" % (n_features, first_feature, best_train.shape, curr_err))
while n_features < 5:
    i = 0
    errs = []

    while i < Xtrain.shape[1]:
        if i not in selected:
            train_x = np.concatenate((best_train, Xtrain[:,i].reshape(-1, 1).reshape(-1, 1)), axis = 1)
            test_x = np.concatenate((best_test, Xval[:,i].reshape(-1, 1).reshape(-1, 1)), axis = 1)
            errs.append(get_mse(train_x, ytrain, test_x, yval))
        elif i in selected:
            errs.append(np.inf)
        i += 1

    curr_feature = np.argmin(errs)
    curr_err = errs[curr_feature]
    best_train = np.concatenate((best_train, Xtrain[:,curr_feature].reshape(-1, 1)), axis = 1)
    best_test = np.concatenate((best_test, Xval[:,curr_feature].reshape(-1, 1)), axis = 1)
    selected.append(curr_feature)
    n_features += 1
    print("N_features: %s, last best feature: %s, current shape: %s, current error: %s" % (n_features, curr_feature, best_train.shape, curr_err))
    
print("Selected features: %s" % ', '.join(str(x) for x in selected))


model = KNeighborsRegressor(n_neighbors=10, algorithm="kd_tree")
model.fit(best_train, ytrain)

print("Starting testing phase")
start = time.time()
data_test = np.genfromtxt("data/test.csv", comments="#", delimiter=",")
Xtest, ytest = data_test[:,:-1], data_test[:,-1]
Xtest = np.take(Xtest, selected, axis=1)
preds = model.predict(Xtest)
mse = mean_squared_error(y_true=ytest, y_pred=preds)
end = time.time()
print("Testing phase completed")
print("Testing mean squared error: %s" % mse)
print("Runtime of testing phase: %s" % (end - start))
