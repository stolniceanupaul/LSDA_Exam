"""
Disclaimer:
This implementation of backward selection makes use of code produced for
the homework assignment number 5. 
"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt

def get_mse(Xtrain, ytrain, Xtrue, ytrue):
    """
    This method returns the mean squared error for the KNN regressor.
    The KNN regressor model is based on the train and test data passed as parameters.
    """
    model = KNeighborsRegressor(n_neighbors=10, algorithm="kd_tree")
    model.fit(Xtrain, ytrain)
    predictions = model.predict(Xtrue)
    mse = mean_squared_error(y_true=ytrue, y_pred=predictions)
    return mse


# load data
print("Loading data ...")

data_train = np.genfromtxt("data/train.csv", comments="#", delimiter=",")
data_val = np.genfromtxt("data/validation.csv", comments="#", delimiter=",")
data_test = np.genfromtxt("data/test.csv", comments="#", delimiter=",")

Xtrain, ytrain = data_train[:, :-1], data_train[:, -1]
Xval, yval = data_val[:, :-1], data_val[:, -1]
Xtest, ytest = data_test[:, :-1], data_test[:, -1]

print("Data loaded!")
print("Bacwkward feature selection started ...")

eliminated_features = []
n_remaining_features = Xtrain.shape[1]
errors = []

while n_remaining_features > 5:
    print("Eliminating %s feature(s)!" % str(15 - n_remaining_features + 1))
    errs = []

    i = 0
    while i < Xtrain.shape[1]:
        if i not in eliminated_features:
            train_x = np.delete(Xtrain, eliminated_features + [i], axis=1)
            val_x = np.delete(Xval, eliminated_features + [i], axis=1)
            errs.append(get_mse(train_x, ytrain, val_x, yval))
        elif i in eliminated_features:
            errs.append(np.inf)
        i += 1

    # Find the feature with the smallest mse
    curr_feature = np.argmin(errs)

    # Get the error for this feature
    curr_err = errs[curr_feature]

    # Add the error to the complete list of errors for removed features
    errors.append(curr_err)
    eliminated_features.append(curr_feature)
    n_remaining_features -= 1

    # Output results
    print("Results:")
    print("N_remaining_features: %s, last eliminated feature: %s, current shape: %s, current error: %s" % (n_remaining_features, curr_feature, train_x.shape, curr_err))
print("Eliminated features: %s" % ', '.join(str(x) for x in eliminated_features))
print("Features left to build the model: %s" % ', '.join(str(x) for x in range(0, 15) if x not in eliminated_features))

print("Backward feature selection completed!")

# Plotting the mean squared error for selecting 5 features based on validation

plt.plot(np.arange(14, 4, -1), errors)
plt.xlim(15, 3)
plt.xlabel('Number of used features in model building')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on validation dataset for different number of features')
plt.xticks(np.arange(14, 4, -1))
plt.show()

# Build the model based on the 5 features
model = KNeighborsRegressor(n_neighbors=10, algorithm="kd_tree")
model.fit(train_x, ytrain)

# Test the model based on the 5 features
print("Testing phase started ...")
start = time.time()
Xtest = np.delete(Xtest, eliminated_features, axis=1)
predictions = model.predict(Xtest)

mse = mean_squared_error(y_true=ytest, y_pred=predictions)
end = time.time()
print("Testing phase completed!")
print("Testing mean squared error: %s" % mse)
print("Runtime of testing phase: %s" % (end - start))
