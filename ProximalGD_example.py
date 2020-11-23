from matplotlib import pyplot as plt
import numpy as np

'''
f(x) = 1/2\|Ax - b\|_2^2 + \lambda\|x\|_1
To reconstruct x0 to x_exact
'''
S = lambda l, v: (v - l)*(v > l) + (v + l)*(v < -l)
m,n = 400,600
A = np.random.randn(m,n)
x_exact = np.zeros((n,1))
ind = np.random.choice(n,100,replace = False)
x0 = np.sign(np.random.randn(n,1))
x_exact[ind] = x0[ind]
b = A @ x_exact + 0.1*np.random.randn(m,1)
lambda_ = 5
loss = lambda x: np.linalg.norm(A@x - b)**2/2 + lambda_*np.linalg.norm(x,ord = 1)
eig_max = np.max(np.linalg.eig(A.T@A)[0].real)
alpha = 1/eig_max
i = 0
pre_x = np.ones((n,1))
x = x0
L = []
while np.linalg.norm(pre_x - x) > 1e-7 and i <= 2000:
    pre_x = x
    g = A.T @ A @ x - A.T @ b
    y = x - alpha * g
    x = S(lambda_*alpha, y)
    i += 1
    L.append(loss(x))
print('The loop ends after {} iterations and the loss decreased\
  to {}'.format(i, L[i-1]))
plt.figure(1)
plt.plot(L)
plt.title('objective-iteration curve')
plt.xlabel('iteration')
plt.ylabel('objective')
plt.figure(2)
plt.plot(range(n),x)
plt.title('x')
plt.xlabel('dimension')
plt.ylabel('entry')
plt.figure(3)
plt.plot(range(n),x_exact)
plt.title('x_exact')
plt.xlabel('dimension')
plt.ylabel('entry')