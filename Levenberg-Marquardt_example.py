import numpy as np
from matplotlib import pyplot as plt


'''
f(x) = 1/2\sum_{i = 1}^5 r_i(x)^2
r_1(x) = x_1^2 + x_2^2 + x_3^2 - 1
r_2(x) = x_1^2 + x_2^2 + (x_3 - 2)^2 - 1
r_3(x) = x_1 + x_2 + x_3 - 1
r_4(x) = x_1 + x_2 - x_3 + 1
r_5(x) = x_1^3  + 3x_2^2 + (5x_3 - x_1 + 1)^2 - 36
'''

def Jac(x):
    J = np.array([[2*x[0],2*x[1],2*x[2]],
                  [2*x[0],2*x[1],2*(x[2]-3)],
                  [1, 1, 1],
                  [1, 1,-1],
                  [3*x[0]**2+2*x[0]-10*x[2]-2,6*x[1],50*x[2]-10*x[0]+10]])
    return J

def res(x):
    r = np.array([(x**2).sum()-1,
                  x[0]**2+x[1]**2+(x[2]-2)**2-1,
                  x.sum()-1,
                  x[0]+x[1]-x[2]+1,
                  x[0]**3+3*x[1]**2+(5*x[2]-x[0]+1)**2-36])
    return r
f = lambda x: np.linalg.norm(res(x))**2/2

gamma = 2
omiga_l, omiga_h = 1/4, 3/4
x = np.zeros(3)
lambda_ = 1
J = Jac(x)
r = res(x)
I = np.eye(3)
i = 0
loss = []
while np.linalg.norm(J.T@r) > 1e-8:
    s = np.linalg.solve(J.T@J + lambda_*I, -J.T@r)
    rho = (f(x) - f(x+s))/(lambda_*s@s + s@J.T@J@s/2)
    if rho > omiga_l:
        x += s
        lambda_ /= gamma
    elif rho > omiga_l:
        x += s
    else:
        lambda_ *= gamma
    
    J = Jac(x)
    r = res(x)
    
    i = i+1
    loss.append(f(x))

print(x)
print(i)
plt.figure()
plt.plot(loss)
plt.xlabel('iter')
plt.ylabel('loss')
plt.title('loss vs iteration curve')