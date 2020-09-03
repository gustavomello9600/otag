from typing import Dict, Tuple
from dataclasses import dataclass

from sympy import diff, sqrt, symbols, simplify, Matrix, Symbol, MatrixSymbol

from suporte.elementos_finitos import Elemento, KeBase, MatrizSimbólica


@dataclass
class MembranaQuadrada(Elemento):

    def traçar_bordas(self) -> None:
        self.bordas = tuple(
                        frozenset({nó, self.nós[i + 1 if i < 3 else 0]})
                        for i, nó in enumerate(self.nós)
                      )

    def __str__(self):
        borda_superior = f"({self.nós[0].x}, {self.nós[0].y}) — ({self.nós[1].x}, {self.nós[1].y})"
        borda_inferior = f"({self.nós[3].x}, {self.nós[3].y}) — ({self.nós[2].x}, {self.nós[2].y})"
        meio = "|" + (max([len(borda_superior), len(borda_inferior)]) - 2) * " " + "|"
        return "\n".join([borda_superior, meio, borda_inferior])


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
                N[(i, k)] = simplify((diff(Ns[k], qsi) * diff(qsi_x, i) + diff(Ns[k], eta) * diff(eta_y, i)))

        B_matrix = Matrix([[N[(x, (i // 2) + 1)] if i % 2 == 0 else 0 for i in range(8)],
                           [N[(y, (i // 2) + 1)] if i % 2 == 1 else 0 for i in range(8)],
                           [N[(y if i % 2 == 0 else x, (i // 2) + 1)] for i in range(8)]])

        B = MatrixSymbol("B", 3, 8)

        t, v, E = symbols("t nu E")
        E_m = MatrixSymbol("E", 3, 3)
        E_matrix = simplify((E / (1 - v ** 2)) * Matrix([[1, v,           0],
                                                         [v, 1,           0],
                                                         [0, 0, (1 - v) / 2]]))

        K_matriz = B.T * E_m * B
        K_matriz = K_matriz.subs({E_m: E_matrix, B: B_matrix}).doit()
        K_matriz = t * K_matriz
        K_matriz = K_matriz.integrate((qsi, -1, 1), (eta, -1, 1))

        print("> K(e) base calculado")
        print("-----------------")

        return K_matriz, {"l": l, "t": t, "v": v, "E": E}


K_base = KeMembranaQuadrada.pronta(cache="K_emq_base.b")
