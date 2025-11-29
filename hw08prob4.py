import numpy as np

m = 40
n = 15
np.random.seed(202503)
A = np.random.randn(m, n)
y = np.random.randn(m)

# (a) Projected Gradient Descent for NNLS

def projected_gd(A, y, alpha=1e-3, tol=1e-6, maxiter=5000):
    x = np.zeros(A.shape[1])
    for k in range(maxiter):
        grad = A.T @ (A @ x - y)
        x_new = x - alpha * grad
        x_new = np.maximum(0, x_new)

        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x


# Compute primal solution by projected GD
x_pgd = projected_gd(A, y)



# (c,d,e) Dual projected gradient ascent

def dual_projected_g_ascent(A, y, eta=1e-3, tol=1e-6, maxiter=5000):
    Q = A.T @ A
    Qinv = np.linalg.inv(Q)
    ATy = A.T @ y

    lam = np.zeros_like(ATy)

    for k in range(maxiter):
        grad = -Qinv @ (ATy + lam)           # gradient of d
        lam_new = lam + eta * grad            # gradient ascent
        lam_new = np.maximum(0, lam_new)      # projection lamda is greater than 0

        if np.linalg.norm(lam_new - lam) < tol:
            break
        lam = lam_new

    # Recover primal: x* = (A^T A)^(-1)(A^T y + λ*)
    x = Qinv @ (ATy + lam)
    x = np.maximum(0, x)
    return lam, x


lam_d, x_d = dual_projected_g_ascent(A, y)


# (f) Verify strong duality

# primal objective
primal_val = 0.5 * np.linalg.norm(y - A @ x_d)**2

# compute dual objective:
# d(λ) = -0.5 (A^T y + λ)^T (A^T A)^(-1)(A^T y + λ) + 0.5||y||^2
Q = A.T @ A
Qinv = np.linalg.inv(Q)
z = A.T @ y + lam_d
dual_val = -0.5 * z.T @ (Qinv @ z) + 0.5 * np.dot(y, y)

print("\n===============================")
print("Strong Duality Check:")
print("Primal objective =", primal_val)
print("Dual objective   =", dual_val)
print("Difference       =", abs(primal_val - dual_val))



# (g) Verify KKT Conditions

# Primal feasibility
primal_feasible = np.all(x_d >= -1e-9)

#Dual feasibility
dual_feasible = np.all(lam_d >= -1e-9)

#Complementary slackness
comp_slack = np.linalg.norm(x_d * lam_d)

#Stationarity
stationarity = np.linalg.norm(A.T @ (A @ x_d - y) - lam_d)

print("\n===============================")
print("KKT VERIFICATION")
print("Primal feasible       =", primal_feasible)
print("Dual feasible         =", dual_feasible)
print("Complementary slack   =", comp_slack)
print("Stationarity residual =", stationarity)

