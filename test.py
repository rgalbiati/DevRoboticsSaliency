import numpy as np 

# test = np.array([9,1,3,4,8,7,2,5,6,0])

# temp = np.argpartition(-test, 4)
# result_args = temp[:4]

# print (result_args)


#____________________________

# test = np.array([[9,1,3], [4,8,7], [2,5,6]])

# temp = top = np.argpartition(test, 2, axis=1)[:, :2]

# print (temp)
# print (top)
# result_args = temp[:4]

# print (result_args)
test = np.array([[9,1,3], [4,8,7], [2,5,6]])

# arr = np.arange(100*100*100).reshape(100, 100, 100)


# np.random.shuffle(arr)


indices =  np.argpartition(test.flatten(), -2)[-3:]


np.vstack(np.unravel_index(indices, test.shape)).T

