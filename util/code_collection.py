import numpy as np
from icecream import ic
import time
import numpy as np
import timeit


def rand_broadcasting_limit():  # random select with numpy broadcasting
    j = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [0.0, np.pi]])
    config = np.random.uniform(low=j[:, 0], high=j[:, 1])
    ic(config)
    ic(config.shape)


def np_diff_broadcasting():  # diff column broadcasting
    j = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [0.0, np.pi]])
    df = np.diff(j)
    ic(df)


def shuffling():  # shuffle dataset
    a = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    np.random.shuffle(a)
    ic(a)


def reverse_loop_index():  # Reverse for loop
    n = 10
    for i in range(-1, -n, -1):
        ic(i)


def array_element_condition():  # array get element from condition
    array = np.array([10, 5, 8, 3, 12, 7, 2])
    indices = np.where(array < 6)
    elements = array[indices]
    ic(elements)


def get_top_lowest_smaller_than_threshold():  # get the top lowest element in array that smaller than threshold
    dist = np.random.randint(low=0, high=20, size=(20,))
    indices = np.where(dist < 6)[0]
    sorted_indices = np.argsort(dist[indices])
    top_3_indices = indices[sorted_indices[:3]]
    ic(top_3_indices)
    ic(dist[top_3_indices])


def sort_based_on_column():  # sort based on column [based on the second column (index 1)]
    nbID = np.array([[10, 20], [12, 22], [11, 20]])
    sorted_indices = np.argsort(nbID[:, 1])
    sorted_nbID = nbID[sorted_indices]
    ic(sorted_nbID)


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


def norm_broadcast():
    a = np.full((6, 3), 2)
    ic(a)
    b = np.random.random((6, 1))
    ic(b)
    ic(a - b)
    ic(b - a)
    c = np.linalg.norm(a - b, axis=0)
    ic(c)


def dictionary_config():
    config = {'param1': 'dfdf', 'param2': 'ssss', 'param3': 'fdae'}
    param1 = config.get('param1', 'default_value1')
    param2 = config.get('param2', 'default_value2')
    ic(param1, param2)
    config.pop("param1")
    ic(config)


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


def array_assign_vector_index():
    a = np.empty((6,2))
    ic(a)
    b = np.random.random((6,1))
    ic(b)

    a[:,0, np.newaxis] = b
    ic(a)

if __name__ == "__main__":
    array_assign_vector_index()