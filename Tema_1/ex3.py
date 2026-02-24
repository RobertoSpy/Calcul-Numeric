import math
import random
import time

def reducere_domeniu(x):
    x_norm = x % math.pi
    if x_norm > math.pi / 2:
        x_norm -= math.pi
    elif x_norm < -math.pi / 2:
        x_norm += math.pi
    return x_norm

# Metoda Lentz 
def my_tan_lentz(x, eps=1e-10):
    x = reducere_domeniu(x)
    
    if abs(abs(x) - math.pi / 2) < 1e-15:
        return float('inf') if x > 0 else float('-inf')

    b0 = 0.0
    mic = 1e-12
    f = b0
    if f == 0.0:
        f = mic

    C = f
    D = 0.0
    j = 1
    
    while True:
        if j == 1:
            a = x
            b = 1.0
        else:
            a = -(x * x)
            b = 2.0 * j - 1.0
            
        D = b + a * D
        if D == 0.0:
            D = mic
            
        C = b + a / C
        if C == 0.0:
            C = mic
            
        D = 1.0 / D
        Delta = C * D
        f = Delta * f
        
        j += 1
        
        if abs(Delta - 1.0) < eps:
            break
            
    return f

# Metoda Polinomului Maclaurin
def my_tan_poly(x):
    x = reducere_domeniu(x)
    
    if abs(abs(x) - math.pi / 2) < 1e-15:
        return float('inf') if x > 0 else float('-inf')
    semn = 1 if x >= 0 else -1
    x_abs = abs(x)
    
    inversat = False
    if x_abs > math.pi / 4:
        x_abs = math.pi / 2 - x_abs
        inversat = True
      
    x_2 = x_abs * x_abs
    x_3 = x_2 * x_abs
    x_4 = x_2 * x_2
    x_6 = x_4 * x_2
    
    c1 = 0.33333333333333333
    c2 = 0.133333333333333333
    c3 = 0.053968253968254
    c4 = 0.0218694885361552
    
    rezultat = x_abs + c1 * x_3 + c2 * (x_2 * x_3) + c3 * (x_4 * x_3) + c4 * (x_6 * x_3)
    
    if inversat:
        rezultat = 1.0 / rezultat
        
    return rezultat * semn

# Test
def testeaza_metode():
    N = 10000
    valori = [random.uniform(-math.pi/2 + 1e-6, math.pi/2 - 1e-6) for _ in range(N)]
    
    print(f"Testare pe {N} de valori generate aleator în (-pi/2, pi/2)...\n")
    
    # Test Lentz
    start_lentz = time.time()
    eroare_lentz = 0.0
    for x in valori:
        eroare_lentz += abs(math.tan(x) - my_tan_lentz(x))
    timp_lentz = time.time() - start_lentz
    
    # Test Polinom
    start_poly = time.time()
    eroare_poly = 0.0
    for x in valori:
        eroare_poly += abs(math.tan(x) - my_tan_poly(x))
    timp_poly = time.time() - start_poly
    
    print(f"{'Metoda':<25} | {'Timp Total (s)':<15} | {'Eroare Medie Absoluta'}")
    print("-" * 70)
    print(f"{'Lentz (Fractii Continue)':<25} | {timp_lentz:<15.5f} | {eroare_lentz/N:.2e}")
    print(f"{'MacLaurin (Polinom)':<25} | {timp_poly:<15.5f} | {eroare_poly/N:.2e}")
    
    x_test = math.pi / 3
    print("\nExemplu pentru x = pi/3:")
    print(f"math.tan(x):      {math.tan(x_test)}")
    print(f"my_tan_lentz(x):  {my_tan_lentz(x_test)}")
    print(f"my_tan_poly(x):   {my_tan_poly(x_test)}")


if __name__ == "__main__":
    testeaza_metode()