import numpy as np
from matplotlib import pyplot as plt


'''
Rosenbrock function: 100(y - x^2)^2 + (1 - x)^2
Using Steihuag-CG method finding the approximated solution of the subproblem
In Steihuag-CG method we use BFGS method to update B_k
'''
f = lambda x: 100*(x[1]-x[0]**2)**2+(1-x[0])**2
g = lambda x: np.array([400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2, -200*x[0]**2 + 200*x[1]])
# h = lambda x: np.array([[-400*x[1] + 1200*x[0]**2+2, -400*x[0]],[-400*x[0], 200]])
eta_s, eta_v = 1/4, 3/4
gamma = 2
x = np.zeros(2)
grad = g(x)
# H = h(x)
I = np.eye(2)
i = 1
radius = 1
loss = []
H = np.array([[3.,1.],[1.,2.]])
while np.linalg.norm(grad) > 1e-8:
    z = np.zeros(2)
    r = g(x)
    p = -r
    while np.linalg.norm(r) > 1e-8:
        if p@H@p <= 0:
            l = (-p@z + np.sqrt((p@z)**2 - p@p*(z@z - r**2)))/p@p
            s = z + l*p
            break
        else:
            pre_z = z.copy()
            alpha = r@r/(p@H@p)
            z += alpha*p
            if np.linalg.norm(z) >= radius:
                l = (-p@pre_z + np.sqrt((p@pre_z)**2 - p@p*(pre_z@pre_z - radius**2)))/(p@p)
                s = pre_z + l*p
                break
            pre_r = r.copy()
            r += alpha * H@p
            if np.linalg.norm(r) < 1e-8:
                s = z
                break
            beta = r@r/(pre_r@pre_r)
            p = -r + beta*p
    rho = (f(x) - f(x+s))/(-grad@s - s@H@s/2)

    if rho >= eta_v:
        x += s
        radius *= gamma
    elif rho >= eta_s:
        x += s
    else:
        radius /= gamma
    
    if (g(x) != grad).all():
        y = g(x) - grad
        grad = g(x)
        H += np.outer(y,y)/(s@y) - np.outer(H@s, H@s)/(s@H@s)
    
    i += 1
    loss.append(f(x))

print(x)
print(i)
plt.plot(loss)
plt.xlabel('iter')
plt.ylabel('loss')
plt.title('loss vs iteration curve')