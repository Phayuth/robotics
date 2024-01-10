import numpy as np
from icecream import ic
import time
import timeit
import matplotlib.pyplot as plt

# ============================================ Numpy Related ================================================


def np_rand_broadcasting_limit():  # random select with numpy broadcasting
    j = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [0.0, np.pi]])
    config1 = np.random.uniform(low=j[:, 0], high=j[:, 1])
    config2 = np.array([[np.random.uniform(low=jnt[0], high=jnt[1])] for jnt in j])
    ic(config1)
    ic(config1.shape)


def np_diff_broadcasting():  # diff column broadcasting
    j = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [0.0, np.pi]])
    df = np.diff(j)
    ic(df)


def np_shuffling():  # shuffle dataset
    a = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    np.random.shuffle(a)
    ic(a)


def np_array_slicing_condition():  # array get element from condition
    array = np.array([10, 5, 8, 3, 12, 7, 2])
    indices = np.where(array < 6)
    elements = array[indices]
    ic(elements)


def np_get_top_lowest_smaller_than_threshold():  # get the top lowest element in array that smaller than threshold
    dist = np.random.randint(low=0, high=20, size=(20,))
    indices = np.where(dist < 6)[0]
    sorted_indices = np.argsort(dist[indices])
    top_3_indices = indices[sorted_indices[:3]]
    ic(top_3_indices)
    ic(dist[top_3_indices])


def np_sort_based_on_column():  # sort based on column [based on the second column (index 1)]
    nbID = np.array([[10, 20], [12, 22], [11, 20]])
    sorted_indices = np.argsort(nbID[:, 1])
    sorted_nbID = nbID[sorted_indices]
    ic(sorted_nbID)


def np_norm_broadcast():
    a = np.full((6, 3), 2)
    ic(a)
    b = np.random.random((6, 1))
    ic(b)
    ic(a - b)
    ic(b - a)
    c = np.linalg.norm(a - b, axis=0)
    ic(c)


def np_array_assign_vector_index():
    a = np.empty((6, 2))
    ic(a)
    b = np.random.random((6, 1))
    ic(b)

    a[:, 0, np.newaxis] = b
    ic(a)


def np_polynomial():
    # create polynomial
    deg3 = np.polynomial.Polynomial([3, 2, 4])
    print(f"deg3: {deg3}")
    deg3div1 = deg3.deriv(1)
    print(f"deg3div: {deg3div1}")
    deg3div2 = deg3.deriv(2)
    print(f"deg3div2: {deg3div2}")

    # evalutate value
    t = np.linspace(0, 5)
    y = deg3(t)

    # polyfit
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    polycoef = np.polynomial.polynomial.polyfit(x, y, 3)
    poly = np.polynomial.Polynomial(polycoef)
    print(f"z: {polycoef}")

    xm = np.linspace(0.0, 5.0, 100)
    ym = poly(xm)
    plt.plot(x, y, 'o', label="Original Datapoint")
    plt.plot(xm, ym, 'r--', label="Fitted Data")
    plt.legend()
    plt.show()


def np_time_measurement_creation():
    ts = time.perf_counter_ns()
    a = np.array([1, 3]).reshape(2, 1)
    te = time.perf_counter_ns()
    print(te - ts)  # result 7400

    ts = time.perf_counter_ns()
    a = np.array([[1], [3]])
    te = time.perf_counter_ns()
    print(te - ts)  # result 4600


# ============================================================================================================


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
    config = {'param1': 'dfdf', 'param2': 'ssss', 'param3': 'fdae'}
    param1 = config.get('param1', 'default_value1')
    param2 = config.get('param2', 'default_value2')
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
            return f'\nconfig = {self.config}, hasParent = {True if self.parent != None else False}, NumChild = {len(self.child)}'

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

    def a(a,b,c):
        pass

    args = inspect.signature(a).parameters
    numArgs = len(args)
    ic(args, numArgs)
check_number_input_function()