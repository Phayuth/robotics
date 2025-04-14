# Numpy Note
- Don't keep adding new element to array, instead create a preknown shape array and change element per iteration.
- Avoid For Loop with Broadcasting ```a = 5*np.array([2,3,4])```
- Element wise comparaison ```a == b```
- Get True index ```np.flatnonzero(y==2)```
- Boolean indexing and accessing```b = array > 2, a[b]```
- Use tilt as Not ```~```
- Accessing element at once ```a = b[[1,3,5,...index]]```
- Sort with index ```i = np.argsort(array)```
- Swap row or col ```new = [1,0,2], array[new,:] or array[:,new]```
- Get unique value ```np.unique(array)```
- Use ufunc at ```np.add.at() or np.....at()```
- Get array using condition ```np.where(array<4)```
- Slicing array but preserve shape ```array[0:3, 3, np.newaxis]```
- Take index ```array.take()```


# Python
- Use enumerate for index and element in for ```for i, v in enumerate(array)```
- Use zip for more than 1 list loop ```for a,b in zip(l1,l2)```
- Use generator for lazy to avoid to much computation ```a = (e[1] for e in events if e[0]=="learn")```, careful when use it again because it consumed when use once.
- Use generator ```islice(), pairwise(), takewhile()```


# Reading
- [Numpy Slicing](https://lisaong.github.io/mldds-courseware/01_GettingStarted/numpy-tensor-slicing.slides.html)