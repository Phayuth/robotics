import numpy as np
from icecream import ic
import time
import timeit


def check_min_index_nested_list():  # check for minimum value and index in nested list, and handle empty list
    nestedList = [[2, 43, 68, 3], [86, 23, 67], [2, 3, 76], []]
    minValue = None
    minIndex = None
    for index, sublist in enumerate(nestedList):
        if not sublist:  # Check if the sublist is empty, Skip empty lists
            continue
        sublistMin = min(sublist)
        if minValue is None or sublistMin < minValue:
            minValue = sublistMin
            minIndex = index
    if minIndex is not None:
        print(f"The minimum value is {minValue} and it is found in the sublist at index {minIndex}.")
    else:
        print("No minimum value found in non-empty sublists.")


def elapsed_time_check():  # calculate elapsed time

    def test_time(a):
        time.sleep(a)

    timestart = time.perf_counter_ns()
    for _ in range(6):
        np.random.uniform(low=-np.pi, high=np.pi)
    timeend = time.perf_counter_ns()
    ic(f"Elapsed time = {(timeend - timestart)}")

    executionTime = timeit.timeit(lambda: test_time(0.01), number=10)
    ic(f"Execution time: {executionTime:.6f} seconds")


def dictionary_config():
    config = {"param1": "dfdf", "param2": "ssss", "param3": "fdae"}
    param1 = config.get("param1", "default_value1")
    param2 = config.get("param2", "default_value2")
    ic(param1, param2)
    config.pop("param1")
    ic(config)


def reverse_loop_index():  # Reverse for loop
    n = 10
    for i in range(-1, -n, -1):
        ic(i)


def node_operation():

    class Node:

        def __init__(self, config, parent=None, cost=0.0) -> None:
            self.config = config
            self.parent = parent
            self.child = []
            self.cost = cost

        def __repr__(self) -> str:
            return f"\nconfig = {self.config}, hasParent = {True if self.parent != None else False}, NumChild = {len(self.child)}"

    # root
    node1 = Node(config=[1, 1], parent=None)

    # child of root
    node2 = Node(config=[2, 2], parent=node1)
    node3 = Node(config=[3, 3], parent=node1)

    node1.child.append(node2)
    node1.child.append(node3)

    # child of node2
    node4 = Node(config=[4, 4], parent=node2)
    node2.child.append(node4)

    # child of node3
    node5 = Node(config=[5, 5], parent=node3)
    node3.child.append(node5)

    # child of node5
    node6 = Node(config=[6, 6], parent=node5)
    node5.child.append(node6)

    def get_ancestor_to_depth(node, depth=None):
        ancestor = []

        def get_ancestor_recursive(node, depth):
            if depth is None:  # go all the ways to the root
                if node.parent is not None:
                    ancestor.append(node.parent)
                    get_ancestor_recursive(node.parent, depth)
            elif depth is not None:  # go until specific depth
                if node.parent is not None and depth != 0:
                    depth -= 1
                    ancestor.append(node.parent)
                    get_ancestor_recursive(node.parent, depth)

        get_ancestor_recursive(node, depth)
        return ancestor

    def near_ancestor_to_depth(XNear, depth):
        nearAncestor = []
        for xNear in XNear:
            nearAncestor.extend(get_ancestor_to_depth(xNear, depth))
        nearAncestor = set(nearAncestor)  # remove duplicate, ordered is lost
        nearAncestor = list(nearAncestor)
        return nearAncestor

    t = get_ancestor_to_depth(node6, None)
    print(t)

    b = [node6, node4]
    c = near_ancestor_to_depth(b, None)
    print(c)


def string_index():
    string = "xyz"
    print(f"String len is: {len(string)}")
    print(string[0])
    print(string[1])
    print(string[2])


def string_seq():
    string = "xyz"
    for index, value in enumerate(string):
        print(f"index {index}, value {value}")


def string_seq_reverse():
    string = "xyz"
    for index, value in reversed(list(enumerate(string))):
        print(f"index {index}, value {value}")


def assert_condition_check():
    assert 1 == 1, "Condition"
    assert 1 == 2, "This must be tree"


def lamda_func():
    a = lambda x: x + 2
    ic(a(1))


def check_class_attribute():

    class A:

        def __init__(self) -> None:
            self.value = [5, 3, 54, 23, 4, 5, 1]
            exist = getattr(self, "value")
            if hasattr(self, "tree"):
                print(hasattr(self, "tree"))
            print(hasattr(self, "tree"))
            print(len(exist))

    a = A()


def check_number_input_function():
    import inspect

    def a(a, b, c):
        pass

    args = inspect.signature(a).parameters
    numArgs = len(args)
    ic(args, numArgs)


def function_decorate():

    def interrupt_decorator(sideFunction):
        def decorator(mainFunction):
            def wrapper(*args, **kwargs):
                try:
                    mainFunction(*args, **kwargs)
                except KeyboardInterrupt:
                    sideFunction()

            return wrapper

        return decorator

    def sideFunc():
        print("exiting")

    @interrupt_decorator(sideFunc)
    def mainFunc():
        print("running...")
        for i in range(100):
            print(f"Iteration : {i}")
            time.sleep(1)
        print("finished")

    mainFunc()


def list_modify_with_for():
    a = [123, 32, 13, 2, 2, 432, 4]
    ic(a)
    for i in range(len(a)):
        a[i] = 30
    ic(a)


def memory_address_test():
    gv = 4
    ic(id(gv))

    def aa(bb):
        ic(id(bb))

        cc = bb
        ic(id(cc))

        return cc

    ee = aa(gv)
    ic(id(ee))


def format_string():
    a = [0.321, 4.213, 421.4132]
    print(f"{a[2]:.3f}")


def get_interval():

    i = list(range(0, 38))
    knot = [12, 23, 31, 34, 37]

    intervals = []

    for val in i:
        if val < knot[0]:
            interval = (-float("inf"), knot[0])
            intervals.append(interval)
        else:
            break

    for val in i[len(intervals) :]:
        for j in range(len(knot) - 1):
            if knot[j] <= val < knot[j + 1]:
                interval = (knot[j], knot[j + 1])
                intervals.append(interval)
                break
        else:
            interval = (knot[-1], float("inf"))
            intervals.append(interval)

    print("Intervals for each i value:")
    for i, interval in zip(i, intervals):
        print(f"i={i}: {interval}")


def cartesian_product():
    from itertools import product
    confignum = 6
    shiftedComb = list(product([-1, 0, 1], repeat=confignum))
    print(f"> shiftedComb: {shiftedComb}")
    print(f"> len :{len(shiftedComb)}")


if __name__ == "__main__":
    cartesian_product()
