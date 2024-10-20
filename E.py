import numpy as np
import matplotlib.pyplot as plt
import warnings

plt.rcParams.update(
    {
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 14,
        'legend.fontsize': 14,
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

        ax = fig.add_subplot(gs[(r-10)//10, 0])
        ax.plot(np.linspace(0, 1, n), x, "--o", color=next(color_cycle), 
                label=f'$r = {r}$ $ \ \left ( ||A x - b ||/||b|| = {np.linalg.norm(A @ x - b)/np.linalg.norm(b):.2e} \\right )$')
        ax.set_ylabel('$U$')
        ax.set_ylim(-0.0028, 0.0028)
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
                label=f'$x_{i}$')
        ax.legend(loc='upper left', ncol=1)
        ax.grid()

        LHS = np.linalg.norm(A @ ritz_vecs[:, i] - ritz_vals[i] * ritz_vecs[:, i])
        RHS = np.abs(beta * ys[49, i])
        assert np.isclose(
            LHS, RHS
        ), f"Bound of D1 failed for x_{i} = Q y_{i}\n {LHS} != {RHS}"

    plt.tight_layout()
    plt.savefig('E2.pdf', format='pdf')
    plt.show()