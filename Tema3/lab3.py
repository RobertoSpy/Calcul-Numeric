import numpy as np
import scipy.linalg as la

# FUNCTIE PENTRU REZOLVAREA SISTEMULUI
def substitutie(R, b_transformat):
    n = len(b_transformat)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        suma = np.sum(R[i, i+1:] * x[i+1:])
        # Evitam impartirea la zero
        if abs(R[i, i]) < 1e-14:
            raise ValueError(f"Matrice singulara detectata la linia {i}")
        x[i] = (b_transformat[i] - suma) / R[i, i]
    return x

#ALGORITMUL LUI HOUSEHOLDER 
def householder_qr(A_init, b_init):
    n = len(A_init)
    A = A_init.copy()
    b = b_init.copy()
    
    Qt = np.eye(n)
    
    for r in range(n - 1):
        sigma = np.sum(A[r:, r]**2)
        
        if sigma <= 1e-15:
            continue
            
        
        k = np.sqrt(sigma)
        if A[r, r] > 0:
            k = -k
            
        # Calculam beta
        beta = sigma - k * A[r, r]
        
        u = np.zeros(n)
        u[r:] = A[r:, r]   
        u[r] = A[r, r] - k  
        
        if abs(beta) < 1e-15:
            continue
            
        # Transformam coloanele lui A (pentru j de la r la n-1)
        for j in range(r, n):
            gamma = np.sum(u[r:] * A[r:, j]) / beta
            A[r:, j] = A[r:, j] - gamma * u[r:]
            
        #  Transformam vectorul b
        gamma_b = np.sum(u[r:] * b[r:]) / beta
        b[r:] = b[r:] - gamma_b * u[r:]
        
        #  Transformam matricea Q (construim Qt)
        for j in range(n):
            gamma_q = np.sum(u[r:] * Qt[r:, j]) / beta
            Qt[r:, j] = Qt[r:, j] - gamma_q * u[r:]
            
   
    return A, b, Qt

def main():
    np.set_printoptions(precision=4, suppress=True)

    print("=== TEMA 3: Descompunerea QR (Householder) ===")
    print("1. Rulează exemplul din PDF")
    print("2. Genereaza sistem aleator (Cerinta 6)")
    optiune = input("Alege o optiune (1 sau 2): ").strip()

    if optiune == '1':
        print("\n--- Rulare Exemplu din PDF ---")
        n = 3
        
        A_init = np.array([
            [0.0, 0.0, 4.0],
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0]
        ])
        s = np.array([3.0, 2.0, 1.0])
    else:
        try:
            n = int(input("\nIntroduceti dimensiunea sistemului n (ex: 50): "))
        except ValueError:
            n = 50
            print("Dimensiune invalida. S-a folosit default n = 50.")
            
        print(f"\n--- Generare sistem de ecuatii random de dimensiune {n}x{n} ---")
        A_init = np.random.rand(n, n)
        s = np.random.rand(n)
    
    #  CERINTA 1: Calculam vectorul b folosind formula din laborator 
    b_init = np.zeros(n)
    for i in range(n):
        for j in range(n):
            b_init[i] += s[j] * A_init[i, j]

    if n <= 10:
        print("\nMatricea inițială A:\n", A_init)
        print("\nVectorul soluție exacta (s):\n", s)
        print("\nVectorul termenilor liberi (b = A*s):\n", b_init)

    # CERINTA 3 
    Q_lib, R_lib = la.qr(A_init)
    x_QR = la.solve_triangular(R_lib, np.dot(Q_lib.T, b_init))
    
    R_house, b_transformat, Qt_house = householder_qr(A_init, b_init)
    x_Householder = substitutie(R_house, b_transformat)
    
    print("\n" + "="*40)
    print("[CERINȚA 3] Rezolvarea sistemului Ax = b")
    if n <= 10:
        print("\nMatricea superior triunghiulara R (Householder):\n", R_house)
        print("\nMatricea ortogonală Q transpus (Householder):\n", Qt_house)
        print("\nSoluția x obținuta cu Householder:\n", x_Householder)
        print("\nSoluția x obținuta cu biblioteca SciPy:\n", x_QR)

    dif_solutii = np.linalg.norm(x_QR - x_Householder, 2)
    print(f"\nDiferența || x_QR - x_Householder ||_2 = {dif_solutii:.6e}")
    
    #  CERINTA 4 
    print("\n" + "="*40)
    print("[CERINTA 4] Verificarea Erorilor (< 10^-6):")
    
    err1 = np.linalg.norm(np.dot(A_init, x_Householder) - b_init, 2)
    err2 = np.linalg.norm(np.dot(A_init, x_QR) - b_init, 2)
    norm_s = np.linalg.norm(s, 2)
    err3 = np.linalg.norm(x_Householder - s, 2) / norm_s
    err4 = np.linalg.norm(x_QR - s, 2) / norm_s
    
    print(f"|| A_init * x_Householder - b ||_2 = {err1:.6e}")
    print(f"|| A_init * x_QR - b ||_2          = {err2:.6e}")
    print(f"|| x_Householder - s ||_2 / ||s||_2 = {err3:.6e}")
    print(f"|| x_QR - s ||_2 / ||s||_2          = {err4:.6e}")
    
    # CERINTA 5 
    print("\n" + "="*40)
    print("[CERINTA 5] Calculul Inversei matricei A")
    A_inv_householder = np.zeros((n, n))
    
    for j in range(n):
        b_inv = Qt_house[:, j]
        coloana_x = substitutie(R_house, b_inv)
        A_inv_householder[:, j] = coloana_x
        
    A_inv_bibl = np.linalg.inv(A_init)
    
    if n <= 10:
        print("\nInversa calculata cu Householder:\n", A_inv_householder)
        print("\nInversa calculata cu SciPy:\n", A_inv_bibl)

    err_inv = np.linalg.norm(A_inv_householder - A_inv_bibl)
    print(f"\nNorma || A^-1_Householder - A^-1_bibl || = {err_inv:.6e}")
    
    if all(e < 1e-6 for e in [err1, err2, err3, err4, err_inv]):
        print("\nSUCCES! Toate normele respecta precizia ceruta.")

if __name__ == "__main__":
    main()