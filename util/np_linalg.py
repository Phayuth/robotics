import numpy as np
import timeit

# a = np.array([0,0,1]).reshape(3,1)
# b = np.array([1,0,0]).reshape(3,1)

# c = np.cross(a,b,axisa=0, axisb=0, axisc=0)
# print(f"==>> c.shape: \n{c.shape}")
# print(f"==>> c: \n{c}")


# c = np.cross(a,b,axis=0)
# print(f"==>> c: \n{c}")

# d = 2 @ a 
# print(f"==>> d: \n{d}")



# # shuffle dataset
# a = np.array([[0,0,0],
#              [1,1,1],
#              [2,2,2]])

# np.random.shuffle(a)
# print(f"==>> b: \n{a}")


# def is_linear_independent(G):
#     det = np.linalg.det(G)
#     if det != 0:
#         print("It is linearly independent")
#     else:
#         print("It is not linearly independent")


# V = np.array([[5, 7], [1, 9]])
# VT = np.transpose(V)

# P = np.array([[1, 1], [2, 2]])
# PT = np.transpose(P)

# G = V @ VT
# GG = P @ PT

# print(G)
# is_linear_independent(G)

# print(GG)
# is_linear_independent(GG)

# print(np.linalg.matrix_rank(G))
# print(np.linalg.matrix_rank(GG))


# # Reverse for loop
# n = 10

# for i in range(-1, -n, -1):
#     print(i)


# nbID = np.array([[10, 20], [12, 22], [11, 20]])

# # Sort the array based on the second column (index 1)
# sorted_indices = np.argsort(nbID[:, 1])

# # Sort the array using the sorted indices
# sorted_nbID = nbID[sorted_indices]

# print("Original array:")
# print(nbID)

# print("Array sorted by the second column:")
# print(sorted_nbID)

# first2index = sorted_nbID[0:2,0]
# print(f"==>> first2index: \n{first2index}")


# n = [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]
# df = np.diff(n)
# print(f"==>> df: \n{df}")
# print(f"==>> n: \n{n}")

# def find_elements_smaller_than_value(input_array, threshold):
#     # Create a Boolean mask for elements smaller than the threshold
#     mask = input_array < threshold
#     print(f"==>> mask: \n{mask}")

#     # Use np.where to find the indices where the mask is True
#     indices = np.where(mask)
#     print(f"==>> indices: \n{indices}")

#     # Use fancy indexing to get the elements at those indices
#     elements = input_array[indices]
#     print(f"==>> elements: \n{elements}")

#     # Combine indices and elements into a list of tuples
#     result = list(zip(indices[0], elements))

#     return result

# # Example usage:
# my_array = np.array([10, 5, 8, 3, 12, 7, 2])
# threshold_value = 6

# elements_below_threshold = find_elements_smaller_than_value(my_array, threshold_value)
# print(elements_below_threshold)



# # a = [[2, 43, 68, 3], [86, 23, 67], [2, 3, 76], []]
# a = [[],[]]

# # Initialize variables to store the minimum value and its index
# min_value = None
# min_index = None

# for index, sublist in enumerate(a):
#     # Check if the sublist is empty
#     if not sublist:
#         continue  # Skip empty lists
    
#     # Find the minimum value in the sublist
#     sublist_min = min(sublist)
    
#     # Check if it's the first minimum found or smaller than the current minimum
#     if min_value is None or sublist_min < min_value:
#         min_value = sublist_min
#         min_index = index

# if min_index is not None:
#     print(f"The minimum value is {min_value} and it is found in the sublist at index {min_index}.")
# else:
#     print("No minimum value found in non-empty sublists.")


# import time

# lines_to_print = ["Line 1", "Line 2", "Line 3", "Line 4"]

# for line in lines_to_print:
#     # Pad the line with spaces to clear previous lines
#     padded_line = line.ljust(max(len(line), len(max(lines_to_print, key=len))))
#     print(f"\r{padded_line}", end='', flush=True)
#     time.sleep(1)

# # Print a newline character to move to the next line after the loop
# print("\nAll done!")

class test:
    @staticmethod
    def adder(a):
        return a + 1
    
testint = test()

b = 4
c = testint.adder(b)
print(f"==>> c: {c}")