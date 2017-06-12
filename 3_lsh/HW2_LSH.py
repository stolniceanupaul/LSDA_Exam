import numpy as np
import pdb
from sklearn.metrics import pairwise
from scipy.sparse import csr_matrix, find
import matplotlib.pyplot as plt
import random
import time
#import math
import matplotlib.pyplot as plt
#from hashlib import sha1
#------------------------------------------------------------------------------
def getPrime(): return 9973
#------------------------------------------------------------------------------
# Generate a list of 'k' random coefficients for the random hash functions,
# while ensuring that the same value does not appear multiple times in the 
# list.
def pickRandomCoeffs(k):
    
    # Generate random integers in range {0, ..., maxInt}
    maxInt = 2**16-1
    
    # Create a list of 'k' random values.
    randList = []
  
    while k > 0:
        # Get a random shingle ID.
        randIndex = random.randint(0, maxInt) 
  
        # Ensure that each random number is unique.
        while randIndex in randList:
            randIndex = random.randint(0, maxInt) 
    
        # Add the random number to the list.
        randList.append(randIndex)
        k = k - 1
    
    return randList
#------------------------------------------------------------------------------
def generateMinHash(k):

    global D
    global N
    global X
    
    maxInt = 2**16 - 1

    # Our random hash function will take the form of:
    #   h(x) = (a*x + b) % c
    # Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
    # a large prime number 
    
    coeffA = pickRandomCoeffs(k)
    coeffB = pickRandomCoeffs(k)
        
    minHash = np.zeros((N, k), dtype=np.int)
    for i in range(N):
             
        # Get doc in sparse format
        doc = X.getrow(i)
      
        # Get index and value of nonzero entries
        # row_idx is always 0 since the matrix has only one row
        # col_idx is the column indices
        # val is the value
        row_idx, col_idx, val = find(doc)
      
        # For each of the random hash functions...
        for j in range(k):
          
        
            # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
            # the maximum possible value output by the hash.
            minHashCode = maxInt + 1
        
            # For each shingle in the document...
            for idx in col_idx:
         
                # Evaluate the hash function.
                hashCode = (coeffA[j] * idx + coeffB[j]) % getPrime()
          
                # Track the lowest hash code seen.
                if hashCode < minHashCode:
                    minHashCode = hashCode
    
            # Add the smallest hash code value as component number 'idx' of the signature.
            minHash[i][j] = minHashCode
      
#        print(i)
    
    return minHash
    
#------------------------------------------------------------------------------
# Preprocess the data
#------------------------------------------------------------------------------
data = np.loadtxt("docword.kos.txt", delimiter = ' ', skiprows = 3)

# Preprocess the data
data[:, 0] = data[:, 0].astype(int) - 1 # Python starts the index from 0
data[:, 1] = data[:, 1].astype(int) - 1 # Python starts the index from 0
data[:, 2] = 1

# Convert to sparse csr_matrix (support row slicing)
# We can use other sparse format, such as coo_matrix, csc_matrix
X = csr_matrix((data[:, 2].astype(int), (data[:, 0] , data[:, 1])))
print(X.get_shape())
(N, D) = X.get_shape()

#------------------------------------------------------------------------------
# Task 1: Bruteforce Jacard similarity
#------------------------------------------------------------------------------
t0 = time.time()

jaccard = X.dot(X.T).todense().astype(float)

#pdb.set_trace()
for i in range(N):
    for j in range(N):
        
        if j >= i:            
                    
            # Compute the set size
            set1 = X.getrow(i).getnnz()
            set2 = X.getrow(j).getnnz()
            
            # Compute Jaccard
            jaccard[i, j] = jaccard[i, j] / (set1 + set2 - jaccard[i, j])
            
        else:
            
            jaccard[i, j] = jaccard[j, i]
        
#    print(i)

# Calculate the elapsed time (in seconds)
elapsed = time.time() - t0
print ("Bruteforce computation took ", elapsed, "sec")
np.savetxt("Bruteforce.txt", jaccard, delimiter = '\t')

print("Average Jaccard similarity: ", np.sum(jaccard) / (N * N))

#------------------------------------------------------------------------------
# Task 2 Compute MinHash
#------------------------------------------------------------------------------
# Time this step.
t0 = time.time()
MinHash = generateMinHash(100)

# Calculate the elapsed time (in seconds)
elapsed = time.time() - t0
print ("Generating MinHash signatures took ", elapsed, "sec")

#------------------------------------------------------------------------------
# Task 3: Generate MinHash for each document and execute bruteforce computation
# Note that the MAE value of different runs are different since it is randomzed algorithm
# So we expect the MAE is reduced when K increases
#------------------------------------------------------------------------------
T = 10
bruteforce = np.loadtxt("Bruteforce.txt")
MAE = [0] * T
for i in range(T):
    
    K = 10 * (i + 1) # ranging in {10, 20, ..., 100}

    # Generate MinHash, a array of N x K
    MinHash = generateMinHash(K)
    
    # Time this step.
    t0 = time.time()
    # Compute normalized Hamming distance = dist / K
    jaccard = 1 - pairwise.pairwise_distances(MinHash, MinHash, 'hamming')

    # Calculate the elapsed time (in seconds)
    elapsed = time.time() - t0
    print ("Computing all pair with MinHash took ", elapsed, "sec with ", K, " hash functions")
    
    # Get average absolute error
    MAE[i] = np.sum(np.absolute(jaccard - bruteforce)) / (N * N)
    
#    print(i)

# Plot to see MAE decreasing when K increases    
plt.plot(MAE, 'r-')
plt.show()
        
#------------------------------------------------------------------------------
# Task 4: Build Locality-sensitive Hashing
#------------------------------------------------------------------------------
K = 100

# After computing 1 - (1 - 0.6^R)^B >= 0.9, we can achieve R = 4 and B = 25
# Note that you can use R <= 4 but B will increase and the space usage also increases
# So B = 4 is somewhat optimal regarding the space usage given K = 100
# But in practice, choosing R = 5 sometime achieves false negatives less then 10% with luck
# and false positive 89%.
# Probability a far away pair in candidate pais is 0.006
    
R = 4 # Number of min hash values in a band
B = int(K / R) # number of hash tables (number of bands)

# Generate MinHash, a array of N x K
MinHash = generateMinHash(K)

# Generate random integers to compute hash value for LSH
random_list = np.random.random_integers(0, getPrime(), R)

# Create B empty tables
M = 20000
TABLEs = np.empty( (B, M), dtype=object)   

# For each table
for i in range(B):
    
    # Compute the index to get MinHash values
    idx1 = R * i
    idx2 = idx1 + R
    
    # For each document, hash it into the table
    for j in range(N):
        
        # Get a string from all values in the band
        minHashValues = MinHash[j][idx1 : idx2]
        
        # The hash value is simply dot product and modulo with N since each table has N buckets
        hashValue = np.dot(minHashValues, random_list) % M
        
        # If bucket is empty, creat a set() and insert document ID into set
        if not TABLEs[i][hashValue]:
            
            # bucket is a set
            TABLEs[i][hashValue] = list()
            TABLEs[i][hashValue].append(j)
           
        else:    
            # Insert document ID into the initialized bucket
            TABLEs[i][hashValue].append(j) # insert into bucket
        
#------------------------------------------------------------------------------
# Task 4: Verify the result
# Note that the false negatives, false positives, and 
# probability of the event that a far away pair has to be checked is not the same on each run.
# It is just approximate the result we have :-)        
#------------------------------------------------------------------------------
# similar pair threshold
J1 = 0.6
# far away pair threshold
J2 = 0.3

# Histogram to identify similar pair using MinHash
minHash_TruePair = np.zeros((N, N), dtype=np.int) # can be replaced by boolean

# Histogram to identify far away pair using MinHash
minHash_FarAwayPair = np.zeros((N, N), dtype=np.int) # can be replaced by boolean

# Histogram to identify far away pair using MinHash
minHash_AllPair = np.zeros((N, N), dtype=np.int) # can be replaced by boolean

# For each table
for i in range(B):
    
    # For each bucket
    for j in range(M):
        
        # If the table has some entries
        if TABLEs[i][j]:
    
            candidate = TABLEs[i][j]
            bucket_size = len(candidate)
            
            # There should be more than 1 document in a bucket
            if bucket_size > 1:
                
                # Loop all pairs in a bucket (a naive approach)
                # We can do better since candidate is a sorted list
                # and each pair is counted twice: (d1, d2) and (d2, d1) 
                for k in range(bucket_size):
                    for l in range(bucket_size):
                                                
                        if l > k:

                            # Get the document ID                              
                            idx1 = candidate[k]  
                            idx2 = candidate[l]
                            
                            # Note that any pair (d1, d2) might collide on several hash tables
                            minHash_AllPair[idx1][idx2] = 1
                            minHash_AllPair[idx2][idx1] = 1
                            
                            # Get true jaccard similarity
                            sim = bruteforce[idx1][idx2]
                            
                            # Verify
                            if sim >= J1:
                                minHash_TruePair[idx1][idx2] = 1
                                minHash_TruePair[idx2][idx1] = 1
                                
                            if sim <= J2:
                                minHash_FarAwayPair[idx1][idx2] = 1
                                minHash_FarAwayPair[idx2][idx1] = 1
                            
                            
print("Candidate size: ", np.count_nonzero(minHash_AllPair) / 2) # dont count each pair twice (d1, d2) = (d2, d1)
print("Number of true pair in the candidate set: ", np.count_nonzero(minHash_TruePair) / 2) # dont count each pair twice (d1, d2) = (d2, d1)

# Number of True Pairs J >= 0.6 without considering identical pairs, e.g., (d1, d1)
numTruePair = np.flatnonzero(bruteforce >= J1).size - N
print("Number of true pairs in bruteforce: ", numTruePair / 2) # dont count each pair twice (d1, d2) = (d2, d1)

# False negative
falseNegative = np.count_nonzero(minHash_TruePair) / numTruePair
print("False negatives: ", 1 - falseNegative)

# False positive
falsePositive = (np.count_nonzero(minHash_AllPair) - np.count_nonzero(minHash_TruePair)) / np.count_nonzero(minHash_AllPair)

print("False positives: ", falsePositive)    

# Number of Far Away Pairs J <= 0.3
numFarAwayPair = np.flatnonzero(bruteforce <= J2).size
colProb_FarAway = np.count_nonzero(minHash_FarAwayPair) / numFarAwayPair

print("Probability that a far away pair is in the candidate set: ", colProb_FarAway)    
