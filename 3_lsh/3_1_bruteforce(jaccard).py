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
# data = csr_matrix((np.ones((no_lines,), dtype=int),  # Fill with ones
#                   (data[:, 1], data[:, 0])),  # For words and documents
#                   dtype=np.int8)  # Using the int datatype

print("-- Reading the data took %s seconds --" % (time.time() - start_time))

# Split the dataset into query set (first 100 documents) and point set (rest of documents)

print("Data loaded and transformed successfully!")
print("Starting bruteforce search of neighbors...")

start_time = time.time()
# Set the similarity threshold
t = 0.8

# Start with the document having docId = 1 from the querry set

# for q_docId in range(1, 51):
#     q = data.getcol(q_docId).toarray()[:, 0]

#     # Store the neighbors and the similarities for each query point
#     neighbor_ids = []
#     similarities = []

#     for x_docId in range(101, no_documents + 1):
#         x = data.getcol(x_docId).toarray()[:, 0]

#         # Calculate the Jaccard similarity to the query point
#         jac_sim = 1 - jaccard(x, q)

#         if not (np.math.isnan(jac_sim)) and jac_sim >= t:
#             print("Querry: %s - Point: %s: Jaccard similarity = %s" % (q_docId, x_docId, jac_sim))
#             neighbor_ids.append(x_docId)
#             similarities.append(jac_sim)
#     # Store the ids of the neighbor documents
#     with open('neighbors_bruteforce.csv', 'ab') as file:
#         rowwriter = csv.writer(file, delimiter=';',
#                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         rowwriter.writerow([q_docId] + [neighbor_ids])

#     # Store the similarities to the neighbor documents
#     with open('similarities_bruteforce.csv', 'ab') as file:
#         rowwriter = csv.writer(file, delimiter=';',
#                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         rowwriter.writerow([q_docId] + [similarities])

# for q in query_set.T:
#     # Start with the document having docId = 100 from the point set
#     p_docId = 100

#     # Store the neighbors and the similarities for each query point
#     neighbor_ids = []
#     similarities = []

#     for x in point_set.T:

#         # Calculate the Jaccard similarity to the query point
#         jac_sim = 1 - jaccard(x, q)
#         print (p_docId, jac_sim)

#         if not (np.math.isnan(jac_sim)) and jac_sim >= t:
#             neighbor_ids.append(p_docId)
#             similarities.append(jac_sim)
#         p_docId += 1

#     # Store the ids of the neighbor documents
#     with open('neighbors_bruteforce.csv', 'ab') as file:
#         rowwriter = csv.writer(file, delimiter=';',
#                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         rowwriter.writerow([q_docId] + [neighbor_ids])

#     # Store the similarities to the neighbor documents
#     with open('similarities_bruteforce.csv', 'ab') as file:
#         rowwriter = csv.writer(file, delimiter=';',
#                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
#         rowwriter.writerow([q_docId] + [similarities])

#     q_docId += 1
#     break
print("-- Brutforce algorithm took %s seconds --" % (time.time() - start_time))