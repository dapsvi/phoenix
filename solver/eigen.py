"""
Eigenvalue solvers for the generalized problem K φ = λ M φ, where K and M are the stiffness and mass matrices of a finite element model.
The eigenvalues λ correspond to squared natural frequencies, and the eigenvectors φ are the corresponding mode shapes.
The smallest eigenvalues are of interest for structural vibration problems, as they correspond to the lowest natural frequencies.
"""

import numpy as np
from scipy.linalg import eigh as dense_eigh


def generalized_eigen(K, M, k=1, max_iter=30, tol=1e-6, verbose=False):
    """Find the k algebraically smallest eigenvalues of  K φ = λ M φ"""
    from pypardiso import spsolve

    n = K.shape[0]
    m = min(k + 4, n)

    rng = np.random.default_rng(42)
    V = rng.standard_normal((n, m))
    V, _ = np.linalg.qr(V)

    eigvals_old = np.full(k, np.inf)

    for it in range(max_iter):
        # shift-invert:  KW  =  MV
        MV = np.asarray(M @ V, dtype=np.float64)
        W = np.empty_like(MV)
        for j in range(m):
            W[:, j] = spsolve(K, MV[:, j])

        K_proj = W.T @ MV
        MW = np.asarray(M @ W, dtype=np.float64)
        M_proj = W.T @ MW

        # enforce symmetry + regularise M_proj
        K_proj = 0.5 * (K_proj + K_proj.T)
        M_proj = 0.5 * (M_proj + M_proj.T)
        M_proj += 1e-12 * np.eye(m)        # regularisation

        # dense eigensolve on the subspace
        try:
            theta, y = dense_eigh(K_proj, M_proj)
        except np.linalg.LinAlgError:
            # fallback: eigenvalues of K_proj alone
            theta, y = dense_eigh(K_proj)

        # sort ascending (smallest eigenvalues first)
        idx = np.argsort(theta)
        theta = theta[idx]
        y = y[:, idx]

        # update subspace
        V = W @ y[:, :m]

        # M-orthogonalise
        MV = M @ V
        M_inner = V.T @ MV
        try:
            L = np.linalg.cholesky(M_inner)
            V = V @ np.linalg.inv(L).T
        except np.linalg.LinAlgError:
            V, _ = np.linalg.qr(V) # fallback

        # convergence check
        if it >= 3:
            change = np.max(np.abs(theta[:k] - eigvals_old))
            if verbose:
                print(f'    eigen iter {it+1:2d}: '
                      f'θ₀={theta[0]:.6e}  Δ={change:.2e}')
            if change < tol:
                break

        eigvals_old = theta[:k].copy()

    return theta[:k], V[:, :k]
