import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import csv


def calc_mean(n, lines_b):
    minhash = open("m_sims_minhash_%s.csv" % n)
    lines = minhash.readlines()

    diffs = []
    for line in range(len(lines_b)):
        diffs.append(abs(float(lines_b[line]) - float(lines[line])))

    return np.mean(diffs)


bruteforce = open("sims_bruteforce.csv")
lines_b = bruteforce.readlines()

mae = []
count = 10
while count <= 100:
    print(count)
    mean = calc_mean(count, lines_b)
    print(mean)
    mae.append(mean)
    count += 10

print(mae)
plt.plot(np.arange(10, 110, 10), mae)
plt.xlabel('Number of permutations')
plt.ylabel('MAE')
plt.title('Mean Average Error and Number of Permutations')
plt.xticks(np.arange(10, 110, 10))
# plt.axis([0, 101, 0, 1])


plt.show()