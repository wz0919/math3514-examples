import numpy as np
from fractions import Fraction
import cvxpy as cvx
import pandas as pd
import random

def fraction_print(A, vec = None):
    '''
    To print the tableau in fraction form
    '''
    if len(A.shape) == 1:
        A = A[None,:]
    m,n = A.shape
    vectors = []
    for i in range(m):
        vector = []
        for j in range(n):
            value = A[i,j]
            vector.append(Fraction(value).limit_denominator())
        vectors.append(vector)
    B = pd.DataFrame(vectors, columns = ['' for i in range(n)])
    if vec != None:
        B.index = [vec]
        print(B)
    else:
        print(B.to_string(index=False))

def simplex(A,b,c,col = None, showstep = True):
    '''
    Simplex method used to check manuallt calculated result of LP
    
    LP: minimize   c^Tx
        subject to Ax = b
                   x \geq 0
    
    In the end use cvxpy to check the calculation of the algrithm
    
    P.S: I print all ratios but only ratios corresponding 
    to positive d_{iq} are valid
    
    Inputs:
    - A: matrix A of shape m*n of LP in standard form
    - b: vector b of shape m of LP in standard form
    - c: vector c of shape n of LP in standard form
    - col: if Basic matrix B's index is known then create tableau 
    with it, else default the last b.shape dimension column (such as 
    phase I process in two-phase simplex method)
    - showstep: boolean value. If True, show every step's calculation
    if False, only show beginning and end
    '''

    n,m = A.shape
    no_solution = False
    solution = np.zeros(m)
    assert (len(b.shape) == 1), 'b shape wrong!'
    assert (len(c.shape) == 1), 'c shape wrong!'
    T1 = np.hstack((A,b[:,None]))
    T2 = np.hstack((c[None,:],np.zeros((1,1))))
    T = np.vstack((T1,T2))
    print('initial tableau:')
    fraction_print(T)

    if col == None:
        col = list(range(m-n,m))

    BMatrix = T[:n,col]
    T[:n,:] = np.linalg.inv(BMatrix)@T[:n,:]
    T[n,:] -= T[n,col]@T[:n,:]
    if showstep == True:
        print('canonical form:')
        fraction_print(T)
    B = col
    N = list(set(list(range(m))) - set(B))
    mu = T[n,N]
    while (mu < 0).any():
        q = N[np.argmin(mu)]
        T[:n,q][abs(T[:n,q])<1e-10] = 0
        if (T[:n,q]<=0).all():
            no_solution = True
            print('no solution!')
            break
        ratio = T[:n,m]/T[:n,q]
        ratio_ = ratio.copy()
        ratio[ratio != ratio] = np.inf
        ratio_[ratio == np.inf] = 100000
        if showstep == True:
            fraction_print(ratio_,'ratio')
        ratio[T[:n,q] < 0] = np.inf
        p = B[np.argmin(ratio)]
        if showstep == True:
            print('p:{},q:{}'.format(p+1,q+1))
        i = np.argmax(T[:n,p])
        T[i,:] /= T[i,q]
        index_no_i = np.delete(np.arange(n+1),i)
        coefficient = T[index_no_i, q]
        T[index_no_i,:] -= np.outer(coefficient, T[i,:])
        if showstep == True:
            fraction_print(T)
        remove_index = B.index(p)
        B[remove_index] = q
        N = list(set(list(range(m))) - set(B))
        mu = T[n,N]
        mu[abs(mu)<1e-10] = 0
    if no_solution == False:
        solution[B] = T[:n,m]
        print('solution by simplex:')
        fraction_print(solution)
        print('minimum:', -Fraction(T[n,m]).limit_denominator())
    else:
        print('minimum: -inf')
    x = cvx.Variable(len(c))
    p = cvx.Problem(cvx.Minimize(c.T@x),
                      [A @ x == b, x >= 0])
    p.solve()
    
    print('By cvxpy:')
    if abs(p.value) < 1e10:
        print('optimal value:', Fraction(p.value).limit_denominator())
        if showstep == True:
            fraction_print(x.value.T, 'p^\\star')
    else:
        print(p.value)
    return T, B

def two_phase(A,b,c, showstep = True):
    '''
    Two- phase simplex method used to check manuallt calculated result of LP
    
    LP: minimize   c^Tx
        subject to Ax = b
                   x \geq 0
    

    P.S: I print all ratios but only ratios corresponding 
    to positive d_{iq} are valid
    
    Inputs:
    - A: matrix A of LP in standard form
    - b: vector b of LP in standard form
    - c: vector c of LP in standard form
    - showstep: boolean value. If True, show every step's calculation
    if False, only show beginning and end
    '''
    
    assert (np.linalg.matrix_rank(A) == A.shape[0]), 'not full rank!'
    sign_b = np.sign(b)
    sign_b[sign_b == 0] = 1
    m,n = A.shape
    A = A*sign_b[:,None]
    b = abs(b)
    A_ = np.hstack((A,np.eye(m)))
    c_ = np.hstack((np.zeros(n),np.ones(m)))
    T,B = simplex(A_, b, c_, showstep = showstep)
    if (np.array(B) < n).all():
        m,n = T.shape
        rest_col = list(range(len(c)))
        A2 = T[:(m-1),rest_col]
        b2 = T[:(m-1),n-1]
        simplex(A2,b2,c,B, showstep = showstep)
    else:
        print('Artificial problem has positive minimum!')
      

# check the algorithm:
row = 4
col = 8
for i in range(10):
    A = np.random.randint(-10,10, size=(row, col))
    b = np.random.randint(-10,10,row)
    c = np.random.randint(-10,10, col)
    two_phase(A,b,c,showstep=False)