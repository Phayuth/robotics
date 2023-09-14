import time
import numpy as np

def sum_loop(a):
    result = 0
    for i in range(10):
        result += a
    return result

if __name__=="__main__":
    timestart = time.perf_counter_ns()
    np.random.uniform(low=-np.pi, high=np.pi)
    np.random.uniform(low=-np.pi, high=np.pi)
    np.random.uniform(low=-np.pi, high=np.pi)
    np.random.uniform(low=-np.pi, high=np.pi)
    np.random.uniform(low=-np.pi, high=np.pi)
    np.random.uniform(low=-np.pi, high=np.pi)
    for _ in range(6):
        np.random.uniform(low=-np.pi, high=np.pi)
    timeend = time.perf_counter_ns()
    print(f"Elapsed time = {(timeend - timestart)}")