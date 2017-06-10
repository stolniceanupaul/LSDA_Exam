import itertools
import numpy as np
import pandas as pd
import scipy
import time


def gen_permutation(n, rows):
    np.random.seed(1)
    return np.array([np.random.permutation(range(1, rows+1)) for i in range(n)])


def gen_signatures(data, n_permutations):
    permutations = gen_permutation(n_permutations, data.shape[1])
    signatures = np.full((n_permutations, len(data)), np.inf).T

    permutations_t = permutations.T

    start_time = time.time()
    for col in range(len(data)):
        for word in range(len(data[col])):
            if data[col][word]:
                for s in range(len(signatures[col])):
                    if signatures[col][s] > permutations_t[word][s]:
                        signatures[col][s] = permutations_t[word][s]
    print("--- Computing signatures for %s permutations took %s seconds ---" % (n_permutations, time.time() - start_time))
    return signatures


def jaccard(col_a, col_b):
    different = len([1 for i in range(len(col_a)) if col_a[i] != col_b[i]])
    return 1.0 * different / len(col_a)


def main(n_permutations):
    print("--- Jaccard distances, %s permutations ---" % n_permutations)
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

    print("Permutations: %s" % n_permutations)
    data = mtx.T
    print("Documents: %s" % len(data))

    signatures = gen_signatures(data, n_permutations)

    dists = []
    sims = []
    start_time = time.time()
    for col_a, col_b in itertools.combinations(signatures, 2):
        # compute jacc dist between two cols
        jac_dist = jaccard(col_a, col_b)
        # if the distance is not NaN...
        if not np.math.isnan(jac_dist):
            # ...add it to the list
            dists.append(jac_dist)
            sims.append(1 - jac_dist)

    print("--- Calculating distances for %s permutations took %s seconds ---" % (n_permutations, time.time() - start_time))

    distance_mean = np.mean(dists)
    similarity_mean = np.mean(sims)
    print("%s permutations - Average Jaccard distance: %s" % (n_permutations, distance_mean))
    print("%s permutations - Average Jaccard similarity: %s" % (n_permutations, similarity_mean))

    np.savetxt(("m_dists_minhash_%s.csv" % n_permutations), dists, fmt='%1.9f', delimiter=",")
    np.savetxt(("m_sims_minhash_%s.csv" % n_permutations), sims, fmt='%1.9f', delimiter=",")

    print("%s,%s,%s" % (n_permutations, distance_mean, similarity_mean))

if __name__ == "__main__":
    main(100)
