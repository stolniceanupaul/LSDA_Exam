import csv
import numpy as np
import time

np.set_printoptions(threshold=np.inf)

print("Loading and splitting data...")
# Read data with from data/docword.enrom.txt
start_time = time.time()

# We split the data into query and points and store them in two dictionaries
query = {}
points = {}

with open('data/docword.enron.txt') as f:
    reader = csv.reader(f, delimiter=' ')
    line_number = 0
    # Keep a set of all the word_ids that are used in the query points
    word_ids = set()
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
            doc_Id = int(row[0])
            word_Id = int(row[1])
            # 100 is the splitting point between query and points
            if doc_Id <= 100:
                word_ids.add(word_Id)
                try:
                    query[doc_Id].append(word_Id)
                except:
                    query[doc_Id] = []
                    query[doc_Id].append(word_Id)
            else:
                try:
                    points[doc_Id].append(word_Id)
                except:
                    points[doc_Id] = []
                    points[doc_Id].append(word_Id)

print("-- Reading the data took %s seconds --" % (time.time() - start_time))

print("Data loaded and transformed successfully!")
print("Starting bruteforce search of neighbors...")

start_time = time.time()
# Set the similarity threshold
t = 0.8

for q_docId in sorted(query.keys()):
    query_words = set(query[q_docId])

    # Store the neighbors and the similarities for each query point
    neighbor_ids = []
    similarities = []

    for x_docId in sorted(points.keys()):
        point_words = set(points[x_docId])
        inter = len(query_words.intersection(point_words))
        union = len(query_words.union(point_words))
        jac_sim = float(inter) / union
        if not (np.math.isnan(jac_sim)) and jac_sim >= t:
            print("Querry: %s - Point: %s: Jaccard similarity = %s" % (q_docId, x_docId, jac_sim))
            neighbor_ids.append(x_docId)
            similarities.append(jac_sim)

    # Store the ids of the neighbor documents
    with open('neighbors_bruteforce.csv', 'ab') as file:
        rowwriter = csv.writer(file, delimiter=';')
        rowwriter.writerow([q_docId] + [neighbor_ids])

    # Store the similarities to the neighbor documents
    with open('similarities_bruteforce.csv', 'ab') as file:
        rowwriter = csv.writer(file, delimiter=';')
        rowwriter.writerow([q_docId] + [similarities])

print("-- Brutforce algorithm took %s seconds --" % (time.time() - start_time))
