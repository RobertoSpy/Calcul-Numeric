import numpy as np
import scipy.linalg as la
import math

def generare_date(n):
  
    
    B = np.random.rand(n, n)
    A = np.dot(B, B.T)
    A += np.eye(n) 
    b = np.random.rand(n)
    
    return A, b

def rezolvare_biblioteca(A, b):
    P, L, U = la.lu(A)
    x_lib = np.linalg.solve(A, b)
    
    return P, L, U, x_lib

def descompunere_ldlt(A, n, eps):
    d = np.zeros(n)
    print(f"\n--- Începe descompunerea LDL^T pentru matrice de dimensiune {n}x{n} ---")
    print("Matricea initiala A:\n", A)
    
    for p in range(n):
        print(f"\n>>> Pasul p = {p}:")
        
        suma_d = 0.0
        for k in range(p):
            suma_d += d[k] * (A[p, k] ** 2)
            
        d[p] = A[p, p] - suma_d
        print(f"  Calcul d[{p}] = element diagonal A[{p},{p}] - suma({suma_d:.4f}) = {d[p]:.4f}")
        
        if abs(d[p]) <= eps:
            raise ValueError(f"Împartire la zero evitata la pasul {p}. Elementul d[{p}] este prea mic.")
            

        for i in range(p + 1, n):
            suma_l = 0.0
            for k in range(p):
                suma_l += d[k] * A[i, k] * A[p, k]
                
            val_veche = A[i, p]
            A[i, p] = (A[i, p] - suma_l) / d[p]
            print(f"  Calcul L[{i}, {p}] = ({val_veche:.4f} - suma({suma_l:.4f})) / d[{p}]({d[p]:.4f}) = {A[i, p]:.4f}")
            
    print("\nFinal descompunere LDL^T")
    print("Vectorul d (diagonala matricei D):\n", d)
    L = np.tril(A, -1) + np.eye(n)
    print("Matricea L (partea inferioara, incluzand 1 pe diagonala):\n", L)
    
    return A, d

def rezolvare_ldlt(A, d, b, n):

    z = np.zeros(n)
    y = np.zeros(n)
    x = np.zeros(n)
    
    print("\n--- Începe rezolvarea sistemului ---")
    print("Rezolvare L * z = b")
    for i in range(n):
        suma = 0.0
        for j in range(i):
            suma += A[i, j] * z[j]  
        z[i] = b[i] - suma          
        print(f"  z[{i}] = b[{i}]({b[i]:.4f}) - suma({suma:.4f}) = {z[i]:.4f}")
        
    print("Vectorul z obtinut:\n", z)
 
    print("\nRezolvare D * y = z")
    for i in range(n):
        y[i] = z[i] / d[i]
        print(f"  y[{i}] = z[{i}]({z[i]:.4f}) / d[{i}]({d[i]:.4f}) = {y[i]:.4f}")
        
    print("Vectorul y obtinut:\n", y)
   
    print("\nRezolvare L^T * x = y")
    for i in range(n - 1, -1, -1):
        suma = 0.0
        for j in range(i + 1, n):
            
            suma += A[j, i] * x[j]
        x[i] = y[i] - suma         
        print(f"  x[{i}] = y[{i}]({y[i]:.4f}) - suma({suma:.4f}) = {x[i]:.4f}")
        
    print("\nVectorul de solutii x final:\n", x)
    return x

def inmultire_Ainit_x(A, x, n):
   
    rezultat = np.zeros(n)
    for i in range(n):
        suma = 0.0
        for j in range(n):
            if j >= i:
               
                suma += A[i, j] * x[j]
            else:
                
                suma += A[j, i] * x[j]
        rezultat[i] = suma
    return rezultat

def main():
   
    try:
        n_val = input("Introduceti dimensiunea sistemului n (ex: 3) (se recomanda valoare mica pentru test vizual): ")
        n = int(n_val) if n_val.strip() else 3
        t_val = input("Introduceti exponentul t pentru precizie epsilon = 10^-t (ex: 8): ")
        t = int(t_val) if t_val.strip() else 9
    except ValueError:
        print("Valori implicite folosite: n=3, t=9 (n e mic pentru a putea vedea pasii pe ecran)")
        n = 3
        t = 9
        
    eps = 10 ** (-t)
    
    print(f"\nGenerare sistem de ecuatii de dimensiune {n}x{n}...")
    A_original, b = generare_date(n)
    
    
    A_bib = A_original.copy()
    A_lucru = A_original.copy()
    
    print("\n1. Se rezolva folosind biblioteca (SciPy/NumPy)...")
    P, L_bib, U_bib, x_lib = rezolvare_biblioteca(A_bib, b)
    print("Descompunere LU (biblioteca) calculata cu succes.")
    print("Matricea de permutare P:\n", P)
    print("Matricea inferior triunghiulara L_bib:\n", L_bib)
    print("Matricea superior triunghiulara U_bib:\n", U_bib)
    print("Vectorul solutie x_lib:\n", x_lib)
    
   
    print("\n2. Se calculeaza descompunerea Choleski LDL^T...")
    A_lucru, d = descompunere_ldlt(A_lucru, n, eps)
    

    det_A = np.prod(d)
    print(f"Determinantul matricei A este: {det_A:.6e}")
    
    print("Se calculează solutia folosind substitutiile...")
    x_chol = rezolvare_ldlt(A_lucru, d, b, n)
    
    print("\n3. Verificarea soluției...")
    
    Ax_chol = inmultire_Ainit_x(A_lucru, x_chol, n)
    
    
    norma1 = np.linalg.norm(Ax_chol - b, 2)
    print(f"|| A_init * x_Chol - b ||_2 = {norma1:.6e}")
    
    norma2 = np.linalg.norm(x_chol - x_lib, 2)
    print(f"|| x_Chol - x_lib ||_2      = {norma2:.6e}")
    
    if norma1 < 10**-8 and norma2 < 10**-8:
        print("\nE bine: Normele sunt în limitele cerute!")
    else:
        print("\nAVERTISMENT: Normele sunt mai mari decât precizia cerută. Verificati condiționarea matricei.")

if __name__ == "__main__":
    main()