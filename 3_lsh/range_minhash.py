from minhash import main as calc_distance
import threading

th = [threading.Thread(target=calc_distance, args=(n,), kwargs={}) for n in range(10, 110, 10)]
[t.start() for t in th]
[t.join() for t in th]
