import numpy as np
from matplotlib import pyplot as plt

'''
minimize   f(x) = \sum_{i = 1}^{n-1}\|x_{i+1} - x_i\|_1
subject to Ax = b

Create x^\star as a piecewise constant signal
reconstruct this x^\star from a random noise initial point
'''

S = lambda l, v: (v - l)*(v > l) + (v + l)*(v < -l)
m,n = 600,1000
D = np.zeros((n-1,n))
for i in range(n-1):
    D[i,i:(i+2)] = np.array([-1,1])
A = np.random.randn(m,n)
x = np.zeros(n)
piece = 10
lenth = int(n/piece)
x = np.zeros(n)
for i in range(piece):
    start = i*lenth
    mu = np.random.randn(1)
    if i == piece - 1:
        x[start:] = mu
    else:
        L = len(x[start:(start+lenth)])
        x[start:(start+lenth)] = mu
b = A@x
x_star = x
plt.title('x_star')
plt.plot(x_star)
x = np.random.randn(n)
z = np.zeros(n-1)
lambda_ = np.zeros_like(b)
mu = np.zeros_like(z)
pre_x = 0
times = 20
t = 1
show = [3,5,10,20]
for i in range(times):
    pre_x = x
    inverse = np.linalg.inv(A.T@A+D.T@D)
    x = inverse@(A.T@(b - lambda_/t) + D.T@(z + mu/t))
    z = S(1/t, D@x - mu/t)
    lambda_ += t*(A@x-b)
    mu += t*(z - D@x)
    if i+1 in show:
        plt.figure()
        plt.title('{} iterations x'.format(i+1))
        plt.plot(x)

for i in range(times):
    pre_x = x
    inverse = np.linalg.inv(A.T@A+D.T@D)
    x = inverse@(A.T@(b - lambda_/t) + D.T@(z + mu/t))
    z = S(1/t, D@x - mu/t)
    lambda_ += t*(A@x-b)
    mu += t*(z - D@x)
plt.figure()
plt.plot(x)