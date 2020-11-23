from matplotlib import pyplot as plt
import numpy as np

'''
f(x,y) = x^4 + y^4 -4xy
initial guess: (-2,3)
method for exact line search: secant method
'''
x,y = -2,3
i = 0
g = np.array([1,1])
grad = lambda x, y: np.array([4*x**3 - 4*y, 4*y**3 - 4*x])
d_phi = lambda x,y,alpha,g: -g @ grad(x-alpha*g[0],y-alpha*g[1])
while np.linalg.norm(g) > 1e-7:
    g = grad(x,y)
    alpha_0 = 0
    alpha_1 = 1
    while abs(alpha_0 - alpha_1) > 1e-8:
        phi_1 = d_phi(x,y,alpha_1,g)
        phi_0 = d_phi(x,y,alpha_0,g)
        alpha = alpha_1 - (alpha_1-alpha_0)*phi_1/(phi_1-phi_0)
        alpha_0 = alpha_1
        alpha_1 = alpha
    x -= alpha_1 * g[0]
    y -= alpha_1 * g[1]
    i += 1
print(i)
print(x,y)