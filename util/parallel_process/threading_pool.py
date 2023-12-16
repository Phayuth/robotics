"""
Threading : https://www.youtube.com/watch?v=IEEhzQoKtQU

"""
import time
import concurrent.futures


def sleeping(y, x):
    print(f"Sleeping for {y} s , input = {x}")
    time.sleep(y)
    return "Done"


#
with concurrent.futures.ThreadPoolExecutor() as executor:
    f1 = executor.submit(sleeping, *(2, "INPUT"))
    f2 = executor.submit(sleeping, *(2, "IIIII"))
    print(f1.result())
    print(f2.result())

#
with concurrent.futures.ThreadPoolExecutor() as executor: # finished NOT order
    inputVar = [(2, "INPUT"), (3, "IIIII")]
    results = [executor.submit(sleeping, *(y, x)) for y, x in inputVar]

    for f in concurrent.futures.as_completed(results):
        print(f.result())

#
with concurrent.futures.ThreadPoolExecutor() as executor: # finished in order
    inputVar = [(2, "INPUT"), (3, "IIIII")]
    results = executor.map(sleeping, *inputVar)

    for result in results:
        print(result)