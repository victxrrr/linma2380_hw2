import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.sparse.linalg import gmres

plt.rcParams.update(
    {
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 14,
        'legend.fontsize': 9,
        'legend.fancybox': False,
        'legend.shadow': True,
        'legend.framealpha': 1,
        'legend.loc': 'lower center',
        'lines.linewidth': 1.05,
        'lines.markersize': 3,
        'grid.linewidth': 0.5,
    }
)

import itertools   
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = itertools.cycle(colors)

def arnoldi(A, b, r):
    """
    Arnoldi's iteration

    @param A: matrix n x n 
    @param b: vector n x 1
    @param r: order of the Krylov subspace

    @return Q: orthonormal basis of the Krylov subspace
    @return H: (augmented) upper Hessenberg matrix (= projection of A onto the Krylov subspace)
    """
    n = len(b)
    Q = np.zeros((n, r)) 
    H = np.zeros((r+1, r))

    Q[:, 0] = b / np.linalg.norm(b)
    for k in range(1, r+1):
        v = A @ Q[:, k-1]
        for l in range(k):
            # Recall that H = Q^T A Q
            # Hence the line below computes H[i, j] = Q[:, i]^T A Q[:, j]
            # and still holds even if we modify v in Q[:, k] directions with k =/= j
            # since the columns of Q are orthonormal and cancel in the projection formula
            H[l, k-1] = np.dot(Q[:, l], v)
            v -= H[l, k-1] * Q[:, l]
        # Now the resulting v corresponds to the component of A Q[:, k-1]
        # in the new basis direction Q[:, k], so its projection is simply its norm
        H[k, k-1] = np.linalg.norm(v)

        if k == r:
            return Q, H
        if not np.isclose(H[k, k-1], 0):
            Q[:, k] = v / H[k, k-1]
        else:
            warnings.warn("Arnoldi's iteration broke down", RuntimeWarning)
            return Q, H
        
def GMRES(A, b, r):
    """
    Generalized Minimal Residual Method

    @param A: matrix n x n 
    @param b: vector n x 1
    @param r: order of the Krylov subspace

    @return x: approximate solution to Ax = b
    """
    Q, H = arnoldi(A, b, r)

    # Now we need to solve (4) using the least squares method 
    RHS = np.zeros(r+1)
    RHS[0] = np.linalg.norm(b)
    y = np.linalg.lstsq(H, RHS)[0]

    return Q @ y

def givens_qr(H, b, r):
    """
    Compute R (tilde) and bhat as described in C2
    """
    R = np.copy(H)
    Omega = np.identity(r+1)
    bhat = np.zeros(r+1)
    bhat[0] = 1
    for i in range(r):
        r = np.sqrt(R[i, i]**2 + R[i+1, i]**2)
        c = R[i, i]/r
        s = R[i+1, i]/r

        R[i, i:], R[i+1, i:] = c * R[i, i:] + s * R[i+1, i:], -s * R[i, i:] + c * R[i+1, i:]
        bhat[i], bhat[i+1] = c * b[i] + s * b[i+1], -s * b[i] + c * b[i+1]
        Omega[i, :], Omega[i+1, :] = c * Omega[i, :] + s * Omega[i+1, :], -s * Omega[i, :] + c * Omega[i+1, :]
    
    bhat = np.linalg.norm(b) * bhat

    assert np.allclose(H, Omega.T @ R) # passed
    return R, bhat

def backward_sub(R, bhat1, r):
    """
    Perform backward substitution as described in C2
    """
    y = np.zeros(r)
    for i in range(r-1, -1, -1):
        sum = 0.0
        for j in range(i+1, r):
            sum += R[i, j] * y[j]
        y[i] = (bhat1[i] - sum)/R[i, i]
    return y

def GMRES_v2(A, b, r):
    Q, H = arnoldi(A, b, r)
    R, bhat = givens_qr(H, b, r)
    y = backward_sub(R, bhat, r)
    
    # print(R.shape)
    # for i in range(R.shape[0]):
    #     for j in range(R.shape[1]):
    #         print(f"{R[i, j]:>8.2e}", end="  ")
    #     print()

    assert np.allclose(y, np.linalg.solve(R[:r,:r], bhat[:r])) # passed
    return Q @ y

def arnoldi_iteration(A, b, n):
    m = A.shape[0]

    h = np.zeros((n + 1, n))
    Q = np.zeros((m, n + 1))

    q = b / np.linalg.norm(b)  # Normalize the input vector
    Q[:, 0] = q  # Use it as the first Krylov vector

    for k in range(n):
        v = A.dot(q)  # Generate a new candidate vector
        for j in range(k + 1):  # Subtract the projections on previous vectors
            h[j, k] = np.dot(Q[:, j], v)
            v = v - h[j, k] * Q[:, j]

        h[k + 1, k] = np.linalg.norm(v)
        eps = 1e-12  # If v is shorter than this threshold it is the zero vector
        if h[k + 1, k] > eps:  # Add the produced vector to the list, unless
            q = v / h[k + 1, k]  # the zero vector is produced.
            Q[:, k + 1] = q
        else:  # If that happens, stop iterating.
            return Q, h
    return Q, h

if __name__ == '__main__':

    ### E1
    n = 101
    h = 1/(n - 1)
    f = lambda xi : -1 if xi <= .2 else 2 if xi >= .8 else 5*xi - 2

    A = np.zeros((n, n))
    diag = np.full(n, -2)
    diag[0] = -1; diag[-1] = -1
    np.fill_diagonal(A, diag)
    np.fill_diagonal(A[1:], -1)
    np.fill_diagonal(A[:, 1:], -1)
    A = A/h**2

    b = np.array([f(i*h) for i in range(n)])
    b[0] = 0; b[-1] = 0

    fig = plt.figure(figsize=(10, 14))
    gs = fig.add_gridspec(5, 1)

    for r in range(10, 60, 10):
        x = GMRES(A, b, r)
        x2 = GMRES_v2(A, b, r)
        xbis = gmres(A, b, maxiter=3)

        ax = fig.add_subplot(gs[(r-10)//10, 0])
        ax.plot(np.linspace(0, 1, n), x, "--o", color=next(color_cycle), 
                label=f'$r = {r}$ $ \ \left ( ||A x - b ||/||b|| = {np.linalg.norm(A @ x - b)/np.linalg.norm(b):.2e} \\right )$')
        ax.plot(np.linspace(0, 1, n), x2, "--o", color=next(color_cycle), 
                label=f'$r = {r}$ (Givens) $ \ \left ( ||A x - b ||/||b|| = {np.linalg.norm(A @ x2 - b)/np.linalg.norm(b):.2e} \\right )$')
        ax.plot(np.linspace(0, 1, n), xbis[0], "--o", color=next(color_cycle),
                label=f'gmres $ \ \left ( ||A x - b ||/||b|| = {np.linalg.norm(A @ xbis[0] - b)/np.linalg.norm(b):.2e} \\right )$')
        ax.set_ylabel('$U$')
        # ax.set_ylim(-0.0028, 0.0028)
        if r == 50:
            ax.set_xlabel('$\\xi$')
        ax.legend()
        ax.grid()
        
    plt.tight_layout()
    plt.savefig('E1.pdf', format='pdf')

    ### E2
    h = 4/(n - 1)
    V = lambda xi : 10 if xi <= -1 or xi >= 1 else 0
    A.fill(0.)
    diag.fill(-2)
    np.fill_diagonal(A, diag)
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:, 1:], 1)
    A = A/h**2

    v = np.array([V(-2 + i*h) for i in range(n)])
    A = -A + np.diag(v)
    b = np.arange(1, n+1)

    Q, H = arnoldi(A, b, 50)
    Qbis, Hbis = arnoldi_iteration(A, b, 49)

    assert np.allclose(Q, Qbis)
    assert np.allclose(H[:50, :49], Hbis)

    beta = H[50, 49]
    H = H[:-1, :]
    eigvals, eigvecs = np.linalg.eig(H)
    sorted_ind = np.argsort(-np.abs(eigvals))[:5]
    ys = eigvecs[:, sorted_ind]
    ritz_vals = eigvals[sorted_ind]
    ritz_vecs = Q @ ys

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(5, 1)

    for i in range(5):
        ax = fig.add_subplot(gs[i, 0])
        ax.stem(np.arange(1, n+1), ritz_vecs[:, i], next(color_cycle), basefmt=" ",
                label=f'$x_{i+1}$')
        ax.legend(loc='upper left', ncol=1)
        ax.grid()

        LHS = np.linalg.norm(A @ ritz_vecs[:, i] - ritz_vals[i] * ritz_vecs[:, i])
        RHS = np.abs(beta * ys[49, i])
        assert np.isclose(
            LHS, RHS
        ), f"Bound of D1 failed for x_{i+1} = Q y_{i+1}\n {LHS} != {RHS}"

    plt.tight_layout()
    plt.savefig('E2.pdf', format='pdf')
    plt.show()