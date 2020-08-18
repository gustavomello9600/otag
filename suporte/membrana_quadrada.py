import pickle
from pathlib import Path

import numpy as np
from sympy import *


class KLocalBase:

    def __init__(self):
        print("-----------------")
        print("> Calculando K(e) base")
        print("Necessário apenas uma vez")

        qsi, eta = symbols("xi eta")

        N1_expr = (1 / 4) * (1 - eta) * (1 - qsi)
        N2_expr = (1 / 4) * (1 + eta) * (1 - qsi)
        N3_expr = (1 / 4) * (1 + eta) * (1 + qsi)
        N4_expr = (1 / 4) * (1 - eta) * (1 + qsi)

        x, y, l, x1, y1 = symbols("x y l x_1 y_1")

        qsi_x = (2 / l) * (x - x1 - l * (sqrt(2) / 2))
        eta_y = (2 / l) * (y - y1 - l * (sqrt(2) / 2))

        N = dict()
        Ns = [0, N1_expr, N2_expr, N3_expr, N4_expr]
        for i in (x, y):
            for k in range(1, 4 + 1):
                N[(i, k)] = simplify((diff(Ns[k], qsi) * diff(qsi_x, i) + diff(Ns[k], eta) * diff(eta_y, i)))

        B_matrix = Matrix([[N[(x, (i // 2) + 1)] if i % 2 == 0 else 0 for i in range(8)],
                           [N[(y, (i // 2) + 1)] if i % 2 == 1 else 0 for i in range(8)],
                           [N[(y if i % 2 == 0 else x, (i // 2) + 1)] for i in range(8)]])

        B = MatrixSymbol("B", 3, 8)

        t, v, E = symbols("t nu E")
        E_m = MatrixSymbol("E", 3, 3)
        E_matrix = simplify((E / (1 - v ** 2)) * Matrix([[1, v, 0],
                                                         [v, 1, 0],
                                                         [0, 0, (1 - v) / 2]]))

        K_matriz = B.T * E_m * B
        K_matriz = K_matriz.subs({E_m: E_matrix, B: B_matrix}).doit()
        K_matriz = t * K_matriz
        K_matriz = K_matriz.integrate((qsi, -1, 1), (eta, -1, 1))

        self.matriz = K_matriz
        self.símbolo_de = {"l": l, "t": t, "v": v, "E": E}

        print("> K(e) base calculado")
        print("-----------------")

    def calcular(self, valor_de):
        parâmetros = valor_de.keys()
        correspondência = {self.símbolo_de[p]: valor_de[p] for p in parâmetros}
        return np.array(self.matriz.evalf(subs=correspondência))


def conseguir_K_local_base():
    arquivo = Path(__file__).parent / "cache" / "K_emq_base.b"

    if not arquivo.exists():
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        K_base = KLocalBase()
        with arquivo.open("wb") as base_binária:
            pickle.dump(K_base, base_binária)
    else:
        with arquivo.open("rb") as base_binária:
            K_base = pickle.load(base_binária)

    return K_base


K_base = conseguir_K_local_base()
