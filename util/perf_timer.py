import time

def sum_loop(a):
    result = 0
    for i in range(10):
        result += a
    return result

if __name__=="__main__":
    timestart = time.perf_counter_ns()
    sum_loop(10)
    timeend = time.perf_counter_ns()
    print(f"Elapsed time = {(timeend - timestart)}")