import numpy as np

# Read the train dataset
train = np.genfromtxt("data/LSDA2017GalaxiesTrain.csv", delimiter=",")

# The train data's target variable (redshifts)
train_y = train[:, -1]

# Calculate the variance of the target variable (redshifts)
variance = np.var(train_y)
print(variance)
