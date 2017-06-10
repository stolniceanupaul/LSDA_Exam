import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

mae = [0.0413203621656, 0.0381935591384, 0.0298611879864, 0.0330207023442, 0.0275645884401,
       0.0344416273363, 0.0308117874964, 0.0351431687536, 0.0347234146752, 0.0305590406641]
plt.plot(np.arange(10, 110, 10), mae) 
plt.xlabel('Number of permutations')
plt.ylabel('Average Similarity')
plt.title('Average Jaccard similarity for different numbers of permutations')
plt.xticks(np.arange(10, 110, 10))
# plt.axis([0, 101, 0, 1])

plt.show()