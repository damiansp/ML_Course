import numpy as np


# Create numpy arrays-----------------------------
mylist = [1., 2., 3., 4.]
mynparray = np.array(mylist)
#print mynparray #  [1. 2. 3. 4.]

one_vector = np.ones(4)
#print one_vector # [1. 1. 1. 1.]

one2Darray = np.ones((2, 4))
#print one2Darray  # [[ 1. 1. 1. 1.]
#                  #  [ 1. 1. 1. 1.]]

zero_vector = np.zeros(4)
#print zero_vector # [ 0. 0. 0. 0.]

empty_vector = np.empty(5)
#print empty_vector
# [  0.00000000e+000   0.00000000e+000   2.12555749e-314   2.12556245e-314
#    2.12556506e-314]

#print mynparray[2] # 3.0

my_matrix = np.array([[1, 2, 3], [4, 5, 6]])
#print my_matrix # [[ 1 2 3]
#                   [ 4 5 6]]
#print my_matrix[1, 2] # 6

#print my_matrix[0:2, 2] # [3 6] # Like R but with 0 idexing and Python seqs
#print my_matrix[0, 0:3] # [1 2 3]
#print my_matrix[0, :]   #  " "

fib_indices = np.array([1, 1, 2, 3])
random_vector = np.random.random(10)
#print random_vector
#[ 0.21729633  0.26606914  0.86814139  0.78804679  0.25289994  0.7896949
#  0.02945564  0.90417739  0.37703658  0.84760072]

#print random_vector[fib_indices]
# [ 0.26606914  0.26606914  0.86814139  0.78804679]

my_vec = np.array([1, 2, 3, 4])
select_index = np.array([True, False, True, False])
#print my_vec[select_index] # [1, 3]

select_cols = np.array([True, False, True])
select_rows = np.array([False, True])
#print my_matrix[select_rows, :] # [[4 5 6]]
#print my_matrix[:, select_cols] # [[ 1 3]
#                                   [ 4 6]]


# Operations on arrays
my_array = np.array([1., 2., 3., 4.])
#print my_array * my_array # item-wise: [1. 4. 9. 16.]
#print my_array ** 2       #  ""  ""     ""
#print my_array - np.ones(4) # [0. 1. 2. 3.]
#print my_array + np.ones(4) # [2. 3. 4. 5.]
#print my_array / 3          # [0.333333 0.666667 1. 1.3333333]
#print my_array / np.array([2., 3., 4., 5.]) # [0.5 0.6666667 0.75 0.8]

#print np.sum(my_array)     # 10.0
#print sum(my_array)        # 10.0
#print np.average(my_array) # 2.5
#print np.mean(my_array)    # 2.5
#print np.sum(my_array) / len(my_array) # 2.5

# Dot Product
array1 = np.array([1., 2., 3., 4.])
array2 = np.array([2., 3., 4., 5.])
#print np.dot(array1, array2) # 40.0
#print np.sum(array1 * array2) # 40.0

array1_mag = np.sqrt(np.dot(array1, array1))
#print array1_mag # 5.4772255...

my_features = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
#print my_features
my_weights = np.array([0.4, 0.5])
#print my_weights
my_preds = np.dot(my_features, my_weights)
#print my_preds # [1.4, 3.2, 5, 6.8]

my_matrix = my_features
my_array = np.array([0.3, 0.4, 0.5, 0.6])
#print np.dot(my_array, my_matrix) # [8.2 10.]

# Matrix multiplication
matrix1 = np.array([[1., 2., 3.], [4., 5., 6.]])
#print matrix1
m2 = np.array([[1., 2.], [3., 4.], [5., 6.]])
#print m2
#print np.dot(matrix1, m2) # [[ 22. 28.]
#                             [ 49. 65.]]



