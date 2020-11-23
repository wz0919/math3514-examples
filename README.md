# math3514: optimization method examples

Examples of some optimization methods learned in MATH3514, Numerical Optimization

`ADMM_example.py`: Reconstructing a piecewise constant signal by **alternating direction method of
multipliers**.

`Gaussian-Newton+WolfeRule_example.py`: Solving a nonlinear least squares problem by **Gaussian-Newton method**. Using **Wofle rule** as the inexact line search's criterion. 

`Levenberg-Marquardt_example.py`: Solving a nonlinear least squares problem by **Levenberg-Marquardt method**.

`ProximalGD_example.py`: Reconstructing a sparse signal by **proximal gradient method**.

`SteepestGD_example.py`: Solving a simple unconstrained minimization problem by **steepest gradient descent method**. Using the **secant method** to do exact line search.

`TrustRegion+CG+BFGS_example.py`: Finding the minimizer of Rosenbrock function by **trust region method**. Using **Steihaug conjugate gradient method** finding the approximated solution of the quadratic subproblem involved in trust region method. Using the BFGS update in **BFGS method** to update B_k.

`Two-phase_Simplex.py`: My implementation of **two-phase simplex method** which I used to check my manually calculated results of LP.
