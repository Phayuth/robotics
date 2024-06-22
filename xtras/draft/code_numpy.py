import numpy as np
from icecream import ic
import time
import matplotlib.pyplot as plt


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


def np_total_distance_node():
    a = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    b = np.diff(a, axis=0)
    c = np.linalg.norm(b, axis=1)
    d = np.sum(c)
    ic(d)

    a = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    c = np.linalg.norm(np.diff(a, axis=0), axis=1)
    d = np.sum(c)
    ic(d)


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
    print(f"polycoef: {polycoef}")

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


def np_daily_random():
    lunch = ["mcdonald",
            "burgerking",
            "momtouch",
            "vietnam food",
            "pasta",
            "yam yam",
            "japanese eel shop",
            "japanese shop 2nd floor",
            "ramen",
            "chicken spicy + rice",
            "korean pork belly and bean",
            "tobokki",
            "taco",
            "pork culet",
            "japanese curry"]
    a = np.random.choice(lunch)
    print(f"> a: {a}")

    buy = ["buy", "not buy"]
    b = np.random.choice(buy)
    print(f"> b: {b}")


def np_logic():
    a = np.array([[1,0,0],
                  [2,0,0]])
    b = np.array([[0,1,1],
                  [1,0,0]])
    c = np.array([[0,0,0],
                  [0,1,1]])
    d = np.logical_or(a,b)
    print(f"> d: {d}")

    e = a | b | c
    print(f"> e: {e}")


def np_clip():
    pos = np.linspace(0, 0.085, 10)
    vel = np.linspace(0.013, 0.100, 10)
    force = np.linspace(5, 220, 10)

    posr = np.clip((3.-230.)/0.085 * pos + 230., 0, 255)
    print(f"> posr: {posr}")

    velr = np.clip(255./(0.1-0.013) * vel - 0.013, 0, 255)
    print(f"> velr: {velr}")

    forcer = np.clip(255./(220.-5.) * force - 5., 0, 255)
    print(f"> forcer: {forcer}")


def np_exist_vector():
    vector = np.array([[1],
                       [2]])
    matrix = np.array([[1, 2, 3, 4, 5, 1, 1, 2, 3, 4],
                       [2, 3, 4, 5, 6, 2, 2, 3, 4, 5]])
    exists = np.all(matrix == vector, axis=0)
    print(f"> exists: {exists}")

    filterout = matrix[:, ~exists]
    print(f"> filterout: {filterout}")


def np_check_column_limit():
    matrix = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

    left_limit = np.array([[1],
                           [2]])

    right_limit = np.array([[5],
                            [6]])

    ll = (matrix >= left_limit)
    print(f"> ll: {ll}")
    rr = (matrix <= right_limit)
    print(f"> rr: {rr}")

    within_limits = np.all(ll & rr, axis=0)
    print(f"> within_limits: {within_limits}")


if __name__ == "__main__":
    np_exist_vector()