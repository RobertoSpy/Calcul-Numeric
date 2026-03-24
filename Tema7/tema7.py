import numpy as np

def schema_horner(coeffs, v):
    n = len(coeffs) - 1
    
    # calcul P(v) si coeficientii polinomului cat Q
    b = coeffs[0]
    q_coeffs = [b]
    for i in range(1, n + 1):
        b = coeffs[i] + b * v
        if i < n:
            q_coeffs.append(b)
    p_v = b
    
    # calcul P'(v) folosind coeficientii lui Q
    d = q_coeffs[0]
    r_coeffs = [d]
    for i in range(1, len(q_coeffs)):
        d = q_coeffs[i] + d * x if 'x' in locals() else q_coeffs[i] + d * v
        if i < len(q_coeffs) - 1:
            r_coeffs.append(d)
    p_prim_v = d
    
    # calcul P''(v) / 2! cu coeficientii lui R
    if len(r_coeffs) > 0:
        h = r_coeffs[0]
        for i in range(1, len(r_coeffs)):
            h = r_coeffs[i] + h * v
        p_secund_v = h * 2
    else:
        p_secund_v = 0
        
    return p_v, p_prim_v, p_secund_v

def metoda_newton(coeffs, x0, eps, k_max):
    xk = x0
    for k in range(k_max):
        p, p_p, _ = schema_horner(coeffs, xk)
        if abs(p_p) <= eps:  # Cazul in care derivata este prea mica
            return None, k
        
        delta_x = p / p_p
        xk = xk - delta_x
        
        if abs(delta_x) < eps:  # Criteriu de oprire
            return xk, k
        if abs(delta_x) > 1e8:  # Criteriu de divergenta
            break
    return None, k_max

def metoda_olver(coeffs, x0, eps, k_max):
    xk = x0
    for k in range(k_max):
        p, p_p, p_s = schema_horner(coeffs, xk)
        if abs(p_p) <= eps:
            return None, k
        
        ck = (p**2 * p_s) / (p_p**3)
        delta_x = (p / p_p) + 0.5 * ck
        xk = xk - delta_x
        
        if abs(delta_x) < eps:
            return xk, k
        if abs(delta_x) > 1e8:
            break
    return None, k_max

def rezolva_tema7(coeffs, eps=1e-10):
    # calcul interval [-R, R] 
    a0 = abs(coeffs[0])
    A_val = max(abs(c) for c in coeffs[1:])
    R = (a0 + A_val) / a0
    
    radacini_distincte = []
    k_max = 1000
    
    # generare puncte de start diferite in intervalul [-R, R]
    puncte_start = np.linspace(-R, R, 50)
    
    print(f"Intervalul de cautare: [{-R:.4f}, {R:.4f}]")
    print("-" * 50)
    print(f"{'Metoda':<10} | {'x0':<8} | {'Radacina':<15} | {'Pasi'}")
    print("-" * 50)

    for x0 in puncte_start:
        # rularea ambelor metode pentru comparatie
        r_n, k_n = metoda_newton(coeffs, x0, eps, k_max)
        r_o, k_o = metoda_olver(coeffs, x0, eps, k_max)
        
        # colectarea radacinilor gasite de Olver
        if r_o is not None:
            # filtrare radacini distincte 
            este_noua = True
            for r_existenta in radacini_distincte:
                if abs(r_o - r_existenta) < eps:
                    este_noua = False
                    break
            if este_noua:
                radacini_distincte.append(r_o)
                print(f"{'Olver':<10} | {x0:>8.2f} | {r_o:>15.10f} | {k_o}")

    # salvare rezultate in fisier 
    radacini_distincte.sort()
    with open("rezultate_tema7.txt", "w") as f:
        f.write(f"Radacinile distincte pentru polinomul cu coeficientii {coeffs}:\n")
        for r in radacini_distincte:
            f.write(f"{r:.10f}\n")
            
    return radacini_distincte

if __name__ == "__main__":
    # P(x) = x^3 - 6x^2 + 11x - 6
    coeffs_test = [1.0, -6.0, 11.0, -6.0]
    
    final_res = rezolva_tema7(coeffs_test)
    print("-" * 50)
    print(f"Radacini finale salvate: {final_res}")