import itertools
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import jaccard
import time

# read data and transpose to a matrix
df = pd.read_csv('kos_header.txt', delim_whitespace=True)
# drop col word count (irrelevant for the task)
df = df.drop('count', 1)
# pivot from list to grouped table
df = df.pivot(index='wordID', columns='docID', values='docID')
# fill blanks with zeros
df = df.fillna(value=0)
mtx = df.as_matrix()

# convert to boolean matrix
x, y = scipy.where(mtx > 0)
# each value > 0 becomes 1
mtx[x, y] = 1

# pairwise comparison of jaccard distances between columns
start_time = time.time()

dists = []
sims = []
# create all pairwise combinations using itertools
# mtx.T = transpose so you row through rows (instead of cols)
for col_a, col_b in itertools.combinations(mtx.T, 2):
    # compute jacc dist between two cols
    jac_dist = jaccard(col_a, col_b)
    # if the distance is not NaN...
    if not np.math.isnan(jac_dist):
        # ...add it to the list
        dists.append(jac_dist)
        sims.append(1 - jac_dist)

print("--- Run took %s seconds ---" % (time.time() - start_time))
print("Average Jaccard distance: ", np.mean(dists))
print("Average Jaccard similarity: ", np.mean(sims))


# save file with .9 precision
np.savetxt("dists_bruteforce.csv", dists, fmt='%1.9f', delimiter=",")
np.savetxt("sims_bruteforce.csv", sims, fmt='%1.9f', delimiter=",")
