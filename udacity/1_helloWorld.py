# Numpy & Lists
import numpy as np

a = np.array(['Hello', 'World'])
a = np.append(a, '!')

for i in a:
    print(i)

for idx, val in enumerate(a):
    print(f'Index: {idx}, Value: {val}')

b = np.array([0, 1, 4, 3, 2])

print(f'''~
    Max: {np.max(b)}
    Mean: {np.average(b)}
    Last: {np.argmax(b)}
~''')

c = np.random.rand(3,3)
print(c)

print(a.shape,b.shape,c.shape,sep='\n')