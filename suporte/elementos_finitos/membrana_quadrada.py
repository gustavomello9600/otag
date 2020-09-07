from typing import Dict, Tuple
from dataclasses import dataclass

from sympy import diff, sqrt, symbols, Matrix, Symbol, MatrixSymbol

from suporte.elementos_finitos import Nó, Elemento, KeBase, MatrizSimbólica


@dataclass
class MembranaQuadrada(Elemento):

    nós: Tuple[Nó, Nó, Nó, Nó]

    def traçar_bordas(self) -> None:
        self.bordas = tuple(
                        frozenset({nó, self.nós[i + 1 if i < 3 else 0]})
                        for i, nó in enumerate(self.nós)
                      )

    def __str__(self) -> str:
        superior = "({0.x: >5.2f}, {0.y: >5.2f}) - ({1.x: >5.2f}, {1.y: >5.2f})".format(*self.nós)
        inferior = "({3.x: >5.2f}, {3.y: >5.2f}) - ({2.x: >5.2f}, {2.y: >5.2f})".format(*self.nós)
        meio = "|" + (max([len(superior), len(inferior)]) - 2) * " " + "|"
        return "\n".join([superior, meio, inferior])


class KeMembranaQuadrada(KeBase):

    def construir(self) -> Tuple[MatrizSimbólica, Dict[str, Symbol]]:
        print("-----------------")
        print("> Calculando K(e) base")
        print("(Necessário apenas uma vez)")

        qsi, eta = symbols("xi eta")

        N1_expr = (1 / 4) * (1 - eta) * (1 + qsi)
        N2_expr = (1 / 4) * (1 + eta) * (1 + qsi)
        N3_expr = (1 / 4) * (1 + eta) * (1 - qsi)
        N4_expr = (1 / 4) * (1 - eta) * (1 - qsi)

        x, y, l, x1, y1 = symbols("x y l x_1 y_1")

        qsi_x = (2 / l) * (x - x1 - l * (sqrt(2) / 2))
        eta_y = (2 / l) * (y - y1 - l * (sqrt(2) / 2))

        N = dict()
        Ns = [0, N1_expr, N2_expr, N3_expr, N4_expr]
        for i in (x, y):
            for k in range(1, 4 + 1):
                N[(i, k)] = (diff(Ns[k], qsi) * diff(qsi_x, i) + diff(Ns[k], eta) * diff(eta_y, i))

        B_matrix = Matrix([[N[(x, (i // 2) + 1)] if i % 2 == 0 else 0 for i in range(8)],
                           [N[(y, (i // 2) + 1)] if i % 2 == 1 else 0 for i in range(8)],
                           [N[(y if i % 2 == 0 else x, (i // 2) + 1)] for i in range(8)]])

        B = MatrixSymbol("B", 3, 8)

        t, v, E = symbols("t nu E")
        E_m = MatrixSymbol("E", 3, 3)
        E_matrix = (E / (1 - v ** 2)) * Matrix([[1, v,           0],
                                                [v, 1,           0],
                                                [0, 0, (1 - v) / 2]])

        K_matriz = B.T * E_m * B
        K_matriz = K_matriz.subs({E_m: E_matrix, B: B_matrix}).doit()
        K_matriz = t * K_matriz
        K_matriz = K_matriz.integrate((qsi, -1, 1), (eta, -1, 1))

        print("> K(e) base calculado")
        print("-----------------")

        return K_matriz, {"l": l, "t": t, "v": v, "E": E}


K_base = KeMembranaQuadrada.pronta(cache="K_emq_base.b")
