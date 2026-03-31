import numpy as np
import matplotlib.pyplot as plt

def f_exact(x):
    return x**4 - 12*x**3 + 30*x**2 + 12

x0, xn = 0.0, 2.0
da = 0.0  # f'(0)
db = 8.0  # f'(2)
x_bar = 1.5
m = 3     
n = 10    


np.random.seed(42) 
puncte_interioare = np.sort(np.random.uniform(x0, xn, n-1))
X = np.concatenate(([x0], puncte_interioare, [xn]))
Y = f_exact(X)

print(f"Nodurile generate (X): {np.round(X, 3)}")


def metoda_celor_mai_mici_patrate(X, Y, m):
    n_puncte = len(X)
    B = np.zeros((m + 1, m + 1))
    f_vec = np.zeros(m + 1)
    
    
    for i in range(m + 1):
        for j in range(m + 1):
            B[i, j] = np.sum(X**(i + j))
        f_vec[i] = np.sum(Y * (X**i))
        
    
    a = np.linalg.solve(B, f_vec)
    return a

def horner(a, x_val):
    m = len(a) - 1
    d = a[m]
    for i in range(m - 1, -1, -1):
        d = a[i] + d * x_val
    return d


a_coefs = metoda_celor_mai_mici_patrate(X, Y, m)
Pm_xbar = horner(a_coefs, x_bar)


eroare_xbar_Pm = abs(Pm_xbar - f_exact(x_bar))
eroare_suma_Pm = sum([abs(horner(a_coefs, X[i]) - Y[i]) for i in range(len(X))])

print("\n--- Metoda celor mai mici patrate ---")
print(f"Pm({x_bar}) = {Pm_xbar:.6f}")
print(f"|Pm({x_bar}) - f({x_bar})| = {eroare_xbar_Pm:.6e}")
print(f"Suma erorilor in noduri = {eroare_suma_Pm:.6e}")



def calcul_spline_cubice(X, Y, da, db):
    n = len(X) - 1
    H = np.zeros((n + 1, n + 1))
    f_vec = np.zeros(n + 1)
    h = np.diff(X) # h[i] = x[i+1] - x[i]
    
    
    H[0, 0] = 2 * h[0]
    H[0, 1] = h[0]
    f_vec[0] = 6 * ((Y[1] - Y[0]) / h[0] - da)

    for i in range(1, n):
        H[i, i-1] = h[i-1]
        H[i, i] = 2 * (h[i-1] + h[i])
        H[i, i+1] = h[i]
        f_vec[i] = 6 * ((Y[i+1] - Y[i]) / h[i] - (Y[i] - Y[i-1]) / h[i-1])
        
    
    H[n, n-1] = h[n-1]
    H[n, n] = 2 * h[n-1]
    f_vec[n] = 6 * (db - (Y[n] - Y[n-1]) / h[n-1])
    
    
    A = np.linalg.solve(H, f_vec)
    return A, h

def evalueaza_spline(x_val, X, Y, A, h):
    
    n = len(X) - 1
    i0 = -1
    for i in range(n):
        if X[i] <= x_val <= X[i+1]:
            i0 = i
            break
            
   
    if i0 == -1:
        if x_val < X[0]: i0 = 0
        else: i0 = n - 1

    
    b_i0 = (Y[i0+1] - Y[i0]) / h[i0] - h[i0] * (A[i0+1] - A[i0]) / 6.0
    c_i0 = (X[i0+1]*Y[i0] - X[i0]*Y[i0+1]) / h[i0] - h[i0] * (X[i0+1]*A[i0] - X[i0]*A[i0+1]) / 6.0
    
    
    termen1 = ((x_val - X[i0])**3 * A[i0+1]) / (6.0 * h[i0])
    termen2 = ((X[i0+1] - x_val)**3 * A[i0]) / (6.0 * h[i0])
    Sf_val = termen1 + termen2 + b_i0 * x_val + c_i0
    
    return Sf_val

A_spline, h_spline = calcul_spline_cubice(X, Y, da, db)
Sf_xbar = evalueaza_spline(x_bar, X, Y, A_spline, h_spline)
eroare_xbar_Sf = abs(Sf_xbar - f_exact(x_bar))

print("\n--- Functii Spline Cubice ---")
print(f"Sf({x_bar}) = {Sf_xbar:.6f}")
print(f"|Sf({x_bar}) - f({x_bar})| = {eroare_xbar_Sf:.6e}")



x_grafic = np.linspace(x0, xn, 500)
y_exact = f_exact(x_grafic)
y_mcmm = [horner(a_coefs, xv) for xv in x_grafic]
y_spline = [evalueaza_spline(xv, X, Y, A_spline, h_spline) for xv in x_grafic]

plt.figure(figsize=(10, 6))
plt.plot(x_grafic, y_exact, 'k-', linewidth=2, label='f(x) exact')
plt.plot(x_grafic, y_mcmm, 'r--', label=f'Polinom cel mai mici patrate (m={m})')
plt.plot(x_grafic, y_spline, 'b-.', label='Spline Cubic $C^2$')
plt.plot(X, Y, 'ko', markersize=6, label='Noduri de interpolare')
plt.plot([x_bar], [f_exact(x_bar)], 'gx', markersize=10, label=f'Punctul x_bar={x_bar}')

plt.title("Comparatie: Cele mai mici patrate vs Spline Cubice")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()