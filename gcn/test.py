import numpy as np
x = np.array([[1,2],[3,4]])
print(x)
print(x.min(axis=0))
print(x.min(axis=1))
print(x.sum(axis=0))
print(len(x))

x = [1,2]
x.append([3,4])
x += [3,4]
print(x)