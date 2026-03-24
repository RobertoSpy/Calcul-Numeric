import numpy as np

# cauta cel mai mare element nediagonal pentru a alege pivotul
def _max_offdiag_indices(A):
    n = A.shape[0]
    p, q = 0, 1
    max_val = abs(A[p, q])
    for i in range(n):
        for j in range(i + 1, n):
            v = abs(A[i, j])
            if v > max_val:
                max_val = v
                p, q = i, j
    return p, q, max_val

# implementare metoda Jacobi pentru calculul valorilor si vectorilor proprii
def metoda_jacobi(A_init, eps=1e-10, k_max=10_000):
    A_init = np.array(A_init, dtype=float)
    if A_init.ndim != 2 or A_init.shape[0] != A_init.shape[1]:
        raise ValueError("Jacobi cere o matrice patratica.")
    if not np.allclose(A_init, A_init.T, atol=eps):
        raise ValueError("Jacobi cere matrice simetrica (A = A^T).")

    n = A_init.shape[0]
    A = A_init.copy()
    U = np.eye(n, dtype=float)

    k = 0
    p, q, max_val = _max_offdiag_indices(A)
    while k < k_max and max_val > eps:
        app = A[p, p]
        aqq = A[q, q]
        apq = A[p, q]

        # calcul unghi rotatie si parametri c, s
        alpha = (app - aqq) / (2.0 * apq)
        t = -alpha + (1.0 if alpha >= 0 else -1.0) * np.sqrt(alpha * alpha + 1.0)
        c = 1.0 / np.sqrt(1.0 + t * t)
        s = t * c

        # actualizare elemente matricea A
        for j in range(n):
            if j == p or j == q:
                continue
            apj_old = A[p, j]
            aqj_old = A[q, j]
            A[p, j] = c * apj_old + s * aqj_old
            A[j, p] = A[p, j]
            A[q, j] = -s * apj_old + c * aqj_old
            A[j, q] = A[q, j]

        A[p, p] = app + t * apq
        A[q, q] = aqq - t * apq
        A[p, q] = 0.0
        A[q, p] = 0.0

       
        for i in range(n):
            uip_old = U[i, p]
            uiq_old = U[i, q]
            U[i, p] = c * uip_old + s * uiq_old
            U[i, q] = -s * uip_old + c * uiq_old

        k += 1
        p, q, max_val = _max_offdiag_indices(A)

    lambda_approx = np.diag(A).copy()
    Lambda = np.diag(lambda_approx)
    
    reziduu = np.linalg.norm(A_init @ U - U @ Lambda)

    return {
        "A_final": A, "U": U, "Lambda": Lambda, "lambda": lambda_approx,
        "iteratii": k, "max_offdiag": max_val, "norma_verificare": reziduu,
    }


def sir_cholesky(A_init, eps=1e-10, k_max=1_000):
    A_init = np.array(A_init, dtype=float)
    Ak = A_init.copy()
    
    for k in range(1, k_max + 1):
        try:
            L = np.linalg.cholesky(Ak)
        except np.linalg.LinAlgError:
            raise ValueError("Matricea nu este pozitiv definita.")

        
        Ak_next = L.T @ L
        diff = np.linalg.norm(Ak_next - Ak)
        Ak = Ak_next

        if diff < eps:
            return {"A_final": Ak, "iteratii": k, "diff_final": diff, "convergent": True}

    return {"A_final": Ak, "iteratii": k_max, "diff_final": diff, "convergent": False}


def rezolva_svd(A, eps=1e-12):
    A = np.array(A, dtype=float)
    p, n = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    V = Vt.T

    # calcul rang si conditionare conform PDF
    tol = max(p, n) * np.finfo(float).eps * s[0]
    rang = int(np.sum(s > tol))
    cond = float(s[0] / s[rang - 1]) if rang > 0 else np.inf

    # pseudoinversa moore-penrose 
    SI = np.zeros((n, p), dtype=float)
    for i in range(rang):
        SI[i, i] = 1.0 / s[i]
    AI = V @ SI @ U.T

    # pseudoinversa prin metoda celor mai mici patrate
    AtA = A.T @ A
    AJ = np.linalg.inv(AtA) @ A.T
    
    # norma diferentei dintre cele doua pseudoinverse
    norma_dif = float(np.linalg.norm(AI - AJ, ord=1))

    return {
        "valori_singulare": s, "rang": rang, "cond": cond,
        "AI": AI, "AJ": AJ, "norma_dif": norma_dif
    }

# testare functii cu datele de exemplu din tema
def ruleaza_demo():
    A_sym = np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [2.0, 2.0, 2.0]])
    
    print("--- Rezultate Jacobi ---")
    rez_j = metoda_jacobi(A_sym)
    print(f"Valori proprii: {rez_j['lambda']}")
    print(f"Norma verificare: {rez_j['norma_verificare']}")

    print("\n--- Rezultate Cholesky ---")
    A_spd = A_sym @ A_sym.T + np.eye(3) 
    rez_ch = sir_cholesky(A_spd)
    print(f"Iteratii: {rez_ch['iteratii']}")
    print("Ultima matrice (diagonala):")
    print(np.round(rez_ch['A_final'], 4))

if __name__ == "__main__":
    ruleaza_demo()