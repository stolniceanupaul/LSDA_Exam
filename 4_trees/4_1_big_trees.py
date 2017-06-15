import numpy as np
from sklearn.tree import DecisionTreeClassifier
import time
from scipy.stats import mode

print("Loading and splitting data...")
start_time = time.time()

# For each one of the 10 trees, prepare the subset to build the model.
# To store a uniform random subset without replacement of 10000 items,
# a 10 x 10000 array is used to store the randomly picked numbers
# which will be the line numbers to be selected from the train dataset.

indexes = [[] for i in range(10)]
global_min = 25667780

for i in range(10):
    indexes[i] = sorted(np.random.choice(25667779, 10000, replace=False))
    if (indexes[i][0] < global_min):
        global_min = indexes[i][0]
        index_of_global_min = i

line_number = 0

subsets = [[] for i in range(10)]
chunk_size = 1000

with open("data/landsat_train.csv") as f:
    EOF = False
    iterations = 0
    while not EOF:
        for i in range(chunk_size):
            # Read each line in the chunk
            line = f.readline()

            line_number += 1
            # When the line_number is found as the global minimum
            while (line_number == global_min):
                # Add the line to the corresponding subset
                subsets[index_of_global_min].append(line.strip().split(','))

                # Remove the minimum
                del indexes[index_of_global_min][0]

                # Generate the candidates for calculating a new global minimum
                candidates = []
                for i in range(10):
                    try:
                        candidates.append(indexes[i][0])
                    except IndexError:
                        # If one of the subsets has been already filled
                        candidates.append(25667780)
                # Find the new index for the global minimum
                index_of_global_min = np.argmin(candidates)
                try:
                    # Find the new globalminimum
                    global_min = indexes[index_of_global_min][0]
                except IndexError:
                    # When the file has reached the last line
                    EOF = True
                    break
            if not line:
                EOF = True
                break
        iterations += 1
        if iterations % 1000 == 0:
            # print(iterations * chunk_size)
            print("Reached line %s in training file!" % line_number)

print("-- Reading the data and creating the subsets took %s seconds --" % (time.time() - start_time))


subsets = np.array(subsets)

print("Starting the training of the 10 trees...")
start_time = time.time()
models = []

for subset in subsets:
    # Separate the data and the labels for each of the subsets
    X = subset[:, 1:10].astype(int)
    Y = subset[:, 0].astype(int)

    # Create a decision tree model
    model = DecisionTreeClassifier(criterion="gini", max_depth=None,
                                   min_samples_split=2, max_features=None)
    # Train the model on the train subset
    model.fit(X, Y)
    # Save the model so it can be applied to the test instances
    models.append(model)
print("-- Training the trees took %s seconds --" % (time.time() - start_time))

# Store all the class labels and the number of their occurences, 
# when predictions are performed on the test dataset
classes = {}

print("Starting the testing phase...")
start_time = time.time()

# Read the test file in chunks of 1000 lines and apply the model
# for data coming from all the 1000 lines
line_number = 0
with open("data/landsat_test.csv") as f:
    EOF = False
    iterations = 0
    while not EOF:
        X_Test = []
        for i in range(chunk_size):
            # Read each line in the chunk
            line = f.readline()
            if not line:
                EOF = True
                break
            line = line.strip().split(',')
            line = np.array(line).astype(int)
            X_Test.append(line)
        if X_Test:
            X_Test = np.array(X_Test)
            preds = []

            for model in models:

                preds.append(model.predict(X_Test))

            preds = np.array(preds)
            prediction = mode(preds)[0][0]
            for pred in prediction:
                try:
                    classes[pred] += 1
                except KeyError:
                    classes[pred] = 1
    iterations += 1
    if iterations % 10 == 0:
        # print(iterations * chunk_size)
        print("Reached line %s in test file!" % line_number)

print("-- Training the trees took %s seconds --" % (time.time() - start_time))
for label in classes:
    print("Class %s has been identified %s times." % (label, classes[label]))
