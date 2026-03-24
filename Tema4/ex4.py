import math
import os
import sys


KMAX = 10000
DIVERGENCE_LIMIT = 1e10


def citeste_vector(nume_fisier):
    valori = []
    with open(nume_fisier, "r", encoding="utf-8") as fisier:
        for linie in fisier:
            linie = linie.strip()
            if linie:
                valori.append(float(linie))
    return valori


def incarca_sistem(folder, index):
    d0 = citeste_vector(os.path.join(folder, "d0_" + str(index) + ".txt"))
    d1 = citeste_vector(os.path.join(folder, "d1_" + str(index) + ".txt"))
    d2 = citeste_vector(os.path.join(folder, "d2_" + str(index) + ".txt"))
    b = citeste_vector(os.path.join(folder, "b_" + str(index) + ".txt"))

    n = len(d0)
    if len(b) != n:
        raise ValueError("d0 si b trebuie sa aiba aceeasi dimensiune")

    p = n - len(d1)
    q = n - len(d2)

    if p < 1 or p >= n:
        raise ValueError("d1 are lungime invalida")
    if q < 1 or q >= n:
        raise ValueError("d2 are lungime invalida")

    return {
        "index": index,
        "n": n,
        "p": p,
        "q": q,
        "d0": d0,
        "d1": d1,
        "d2": d2,
        "b": b,
    }


def diagonala_principala_nenula(d0, eps):
    for valoare in d0:
        if abs(valoare) <= eps:
            return False
    return True


def gauss_seidel(sistem, eps):
    d0 = sistem["d0"]
    d1 = sistem["d1"]
    d2 = sistem["d2"]
    b = sistem["b"]
    n = sistem["n"]
    p = sistem["p"]
    q = sistem["q"]

    if not diagonala_principala_nenula(d0, eps):
        return False, 0, math.inf, None, "Exista elemente nule pe diagonala principala."

    x = [0.0] * n

    for iteratia in range(1, KMAX + 1):
        delta = 0.0

        for i in range(n):
            suma = 0.0

            if i - p >= 0:
                suma += d1[i - p] * x[i - p]
            if i + p < n:
                suma += d1[i] * x[i + p]

            if i - q >= 0:
                suma += d2[i - q] * x[i - q]
            if i + q < n:
                suma += d2[i] * x[i + q]

            vechi = x[i]
            x[i] = (b[i] - suma) / d0[i]

            diferenta = abs(x[i] - vechi)
            if diferenta > delta:
                delta = diferenta

        if delta < eps:
            return True, iteratia, delta, x, "Convergenta obtinuta."

        if delta > DIVERGENCE_LIMIT or not math.isfinite(delta):
            return False, iteratia, delta, None, "Metoda Gauss-Seidel a divergenta."

    return False, KMAX, delta, None, "Nu s-a obtinut convergenta in numarul maxim de iteratii."


def inmultire_ax(sistem, x):
    n = sistem["n"]
    p = sistem["p"]
    q = sistem["q"]
    d0 = sistem["d0"]
    d1 = sistem["d1"]
    d2 = sistem["d2"]

    y = [0.0] * n

    for i in range(n):
        y[i] = d0[i] * x[i]

    for i in range(len(d1)):
        j = i + p
        y[i] += d1[i] * x[j]
        y[j] += d1[i] * x[i]

    for i in range(len(d2)):
        j = i + q
        y[i] += d2[i] * x[j]
        y[j] += d2[i] * x[i]

    return y


def norma_inf(v1, v2):
    maxim = 0.0
    for i in range(len(v1)):
        diferenta = abs(v1[i] - v2[i])
        if diferenta > maxim:
            maxim = diferenta
    return maxim


def preview_vector(v, cate=4):
    if len(v) <= 2 * cate:
        return "[" + ", ".join("{:.10f}".format(x) for x in v) + "]"

    inceput = ", ".join("{:.10f}".format(x) for x in v[:cate])
    sfarsit = ", ".join("{:.10f}".format(x) for x in v[-cate:])
    return "[" + inceput + ", ..., " + sfarsit + "]"


def sisteme_disponibile(folder):
    rezultate = []
    for nume in os.listdir(folder):
        if nume.startswith("d0_") and nume.endswith(".txt"):
            index_text = nume[3:-4]
            if index_text.isdigit():
                rezultate.append(int(index_text))
    rezultate.sort()
    return rezultate


def afiseaza_sistem(sistem, rezultat, eps):
    convergent, iteratii, delta, x, mesaj = rezultat

    print("Sistemul", sistem["index"])
    print("  n =", sistem["n"])
    print(
        "  Diagonale secundare: p =",
        sistem["p"],
        "(x =",
        sistem["n"] - 1 - sistem["p"],
        "), q =",
        sistem["q"],
        "(y =",
        sistem["n"] - 1 - sistem["q"],
        ")",
    )
    print(
        "  Toate elementele din d0 sunt nenule:",
        "DA" if diagonala_principala_nenula(sistem["d0"], eps) else "NU",
    )

    if not convergent:
        print("  Gauss-Seidel:", mesaj)
        print("  Iteratii efectuate:", iteratii)
        print("  Delta finala: {:.6e}".format(delta))
        print()
        return

    y = inmultire_ax(sistem, x)
    norma = norma_inf(y, sistem["b"])

    print("  Gauss-Seidel:", mesaj)
    print("  Iteratii:", iteratii)
    print("  Delta finala: {:.6e}".format(delta))
    print("  ||A * xGS - b||_inf = {:.6e}".format(norma))
    print("  xGS (preview) =", preview_vector(x))
    print("  y = A * xGS (preview) =", preview_vector(y))
    print()


def main():
    putere = 8
    sistem_cerut = None

    if len(sys.argv) >= 2:
        putere = int(sys.argv[1])

    if len(sys.argv) >= 4 and sys.argv[2] == "--system":
        sistem_cerut = int(sys.argv[3])

    if putere < 1:
        raise ValueError("Puterea pentru epsilon trebuie sa fie pozitiva")

    eps = 10 ** (-putere)
    folder = os.path.dirname(os.path.abspath(__file__))
    indici = sisteme_disponibile(folder)

    if not indici:
        raise FileNotFoundError("Nu exista fisiere de forma d0_i.txt")

    if sistem_cerut is not None:
        if sistem_cerut not in indici:
            raise ValueError("Sistemul cerut nu exista")
        indici = [sistem_cerut]

    print("Epsilon = 1e-{} = {:.1e}".format(putere, eps))
    print("KMAX =", KMAX)
    print()

    for index in indici:
        sistem = incarca_sistem(folder, index)
        rezultat = gauss_seidel(sistem, eps)
        afiseaza_sistem(sistem, rezultat, eps)


if __name__ == "__main__":
    main()
