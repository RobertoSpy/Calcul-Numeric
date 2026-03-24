import numpy as np
import math


def sigmoid(z):
    z = np.clip(z, -500, 500)
    sig = 1.0 / (1.0 + np.exp(-z))
    sig = np.clip(sig, 1e-15, 1.0 - 1e-15)
    return sig

functii_test = [
    {
        "nume": "1. Funcția Loss Logistic (l)",
       "f": lambda x: -np.log(1 - sigmoid(x[0] - x[1])) - np.log(sigmoid(x[0] + x[1]))+ 0.5 * (x[0]**2 + x[1]**2),
        "grad": lambda x: np.array([
            sigmoid(x[0] - x[1]) + sigmoid(x[0] + x[1]) - 1 + x[0],
            sigmoid(x[0] + x[1]) - sigmoid(x[0] - x[1]) - 1 + x[1]
        ]),
        "start": np.array([1.0, 1.0])
    },
    {
        "nume": "2. F(x1, x2) = x1^2 + x2^2 - 2x1 - 4x2 - 1",
        "f": lambda x: x[0]**2 + x[1]**2 - 2*x[0] - 4*x[1] - 1,
        "grad": lambda x: np.array([2*x[0] - 2, 2*x[1] - 4]),
        "start": np.array([0.0, 0.0])
    },
    {
        "nume": "3. F(x1, x2) = 3x1^2 - 12x1 + 2x2^2 + 16x2 - 10",
        "f": lambda x: 3*x[0]**2 - 12*x[0] + 2*x[1]**2 + 16*x[1] - 10,
        "grad": lambda x: np.array([6*x[0] - 12, 4*x[1] + 16]),
        "start": np.array([0.0, 0.0])
    },
    {
        "nume": "4. F(x1, x2) = x1^2 - 4x1x2 + 4.5x2^2 - 4x2 + 3",
        "f": lambda x: x[0]**2 - 4*x[0]*x[1] + 4.5*x[1]**2 - 4*x[1] + 3,
        "grad": lambda x: np.array([2*x[0] - 4*x[1], -4*x[0] + 9*x[1] - 4]),
        "start": np.array([0.0, 0.0])
    },
    {
        "nume": "5. F(x1, x2) = x1^2*x2 - 2x1*x2^2 + 3x1*x2 + 4",
        "f": lambda x: x[0]**2 * x[1] - 2*x[0] * x[1]**2 + 3*x[0]*x[1] + 4,
        "grad": lambda x: np.array([
            2*x[0]*x[1] - 2*x[1]**2 + 3*x[1],
            x[0]**2 - 4*x[0]*x[1] + 3*x[0]
        ]),
        
        "start": np.array([-0.5, 0.1]) 
    }
]


def gradient_numeric(F, x, h=1e-5):
   
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        
        x1, x2, x3, x4 = x.copy(), x.copy(), x.copy(), x.copy()
        
        x1[i] += 2 * h
        x2[i] += h
        x3[i] -= h
        x4[i] -= 2 * h
        
        F1 = F(x1)
        F2 = F(x2)
        F3 = F(x3)
        F4 = F(x4)
        
        grad[i] = (-F1 + 8*F2 - 8*F3 + F4) / (12 * h)
        
    return grad


def calculeaza_eta_backtracking(F, x, grad_x, beta=0.8):
   
    eta = 1.0
    p = 1
    Fx = F(x)
    norma_grad_patrat = np.linalg.norm(grad_x) ** 2
    
    
    while p < 8:
        x_urmator = x - eta * grad_x
        if F(x_urmator) <= Fx - (eta / 2.0) * norma_grad_patrat:
            break # Am gasit un pas bun, iesim!
            
        eta = eta * beta
        p += 1
        
    return eta


def gradient_descent(F, grad_F, x0, metoda_grad="analitic", metoda_eta="constant", eps=1e-5, kmax=30000):
    x = np.array(x0, dtype=float)
    k = 0
    eta_constant = 1e-3
    
    while True:
       
        if metoda_grad == "numeric":
            grad_x = gradient_numeric(F, x)
        else:
            grad_x = grad_F(x)
            
      
        if metoda_eta == "backtracking":
            eta = calculeaza_eta_backtracking(F, x, grad_x)
        else:
            eta = eta_constant
            
        lungime_pas = eta * np.linalg.norm(grad_x)
        
        
        if lungime_pas <= eps:
            return x, k, "Convergenta"
        if lungime_pas > 1e10:
            return x, k, "Divergenta (Pasul a explodat)"
        if k >= kmax:
            return x, k, "Divergenta (S-a atins kmax)"
            
       
        x = x - eta * grad_x
        k += 1

def main():
    eps = 1e-5
    
    for test in functii_test:
        print("-" * 70)
        print(f"Testam: {test['nume']}")
        print(f"Punct start ales: {test['start']}")
        print("-" * 70)
        
        # Combinatii de testat
        scenarii = [
            ("analitic", "constant"),
            ("numeric", "constant"),
            ("analitic", "backtracking"),
            ("numeric", "backtracking")
        ]
        
        print(f"{'Gradient':<10} | {'Invatare (eta)':<15} | {'Iteratii':<10} | {'Solutie gasita (x1, x2)'}")
        print("-" * 70)
        
        for grad_type, eta_type in scenarii:
            solutie, iteratii, status = gradient_descent(
                test["f"], test["grad"], test["start"], 
                metoda_grad=grad_type, metoda_eta=eta_type, eps=eps
            )
            
            # Formatam output-ul
            if "Divergenta" in status:
                rezultat_str = status
            else:
                rezultat_str = f"({solutie[0]:.5f}, {solutie[1]:.5f})"
                
            print(f"{grad_type:<10} | {eta_type:<15} | {iteratii:<10} | {rezultat_str}")
        print("\n")

if __name__ == "__main__":
    main()