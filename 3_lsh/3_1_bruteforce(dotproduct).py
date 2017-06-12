import csv
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from scipy.spatial.distance import jaccard
import time

np.set_printoptions(threshold=np.inf)

print("Loading and splitting data...")
# Read data with from data/docword.enrom.txt
start_time = time.time()

# First read only the number of documents, words and lines
with open('data/docword.enron.txt') as f:
    reader = csv.reader(f, delimiter=' ')
    line_number = 0
    for row in reader:
        line_number += 1
        if line_number == 1:
            # Read the number of documents from the first line
            no_documents = eval(row[0])

        elif line_number == 2:
            # Read the number of words from the second line
            no_words = eval(row[0])

        elif line_number == 3:
            # Read the total number of lines from the third line
            no_lines = eval(row[0])
        else:
            break

# Read the rest of the data
data = np.loadtxt("data/docword.enron.txt",
                  delimiter=' ', skiprows=3, dtype='int')

# We use a sparse matrix for storing the binary matrix
data = csr_matrix((np.ones((no_lines,), dtype=int),  # Fill with ones
                  (data[:, 1], data[:, 0])),  # For words and documents
                  dtype=np.int8)  # Using the int datatype

print("-- Reading the data took %s seconds --" % (time.time() - start_time))

# Split the dataset into query set (first 100 documents) and point set (rest of documents)

print("Data loaded and transformed successfully!")
print("Starting bruteforce search of neighbors...")

start_time = time.time()
# Set the similarity threshold
t = 0.8

# x = data.getcol(11)
# x_den = x.toarray()[:, 0]
# q = data.getcol(12)
# q_den = q.toarray()[:, 0]
# jac = 1 - jaccard(q_den, x_den)
# print(jac)

# inters = q.transpose().dot(x).todense().astype(float)[0, 0]

# s1 = q.getnnz()

# s2 = x.getnnz()

# j = inters / (s1 + s2 - inters)
# print(j)
# Start with the document having docId = 1 from the querry set

for q_docId in range(1, 101):
    # Store the neighbors and the similarities for each query point
    neighbor_ids = []
    similarities = []

    # Get the document vector for the q_docId query
    q = data.getcol(q_docId)
    for x_docId in range(101, no_documents + 1):
        x = data.getcol(x_docId)
        intersection = q.transpose().dot(x).todense().astype(float)[0, 0]
        set1 = q.getnnz()
        set2 = x.getnnz()

        jac_sim = intersection / (set1 + set2 - intersection)
        if not (np.math.isnan(jac_sim)) and jac_sim >= t:
            print("Querry: %s - Point: %s: Jaccard similarity = %s" % (q_docId, x_docId, jac_sim))
            neighbor_ids.append(x_docId)
            similarities.append(jac_sim)
    # Store the ids of the neighbor documents
    with open('neighbors_bruteforce.csv', 'ab') as file:
        rowwriter = csv.writer(file, delimiter=';',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rowwriter.writerow([q_docId] + [neighbor_ids])

    # Store the similarities to the neighbor documents
    with open('similarities_bruteforce.csv', 'ab') as file:
        rowwriter = csv.writer(file, delimiter=';',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rowwriter.writerow([q_docId] + [similarities])
    break


print("-- Brutforce algorithm took %s seconds --" % (time.time() - start_time))