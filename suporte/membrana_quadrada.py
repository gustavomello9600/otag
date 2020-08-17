from timeit import default_timer

import numpy as np
from sympy import *

matriz_K_está_pronta = False
K_matrix             = None
l, t, v, E           = None, None, None, None


def elemento_de_membrana_quadrada(L, t_=0.01, v_=0.3, E_=210e9):
    global matriz_K_está_pronta
    global K_matrix
    global l, t, v, E

    if not matriz_K_está_pronta:
        print("-----------------")
        print("> Calculando K(e)")
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

        K_matrix = B.T * E_m * B
        K_matrix = K_matrix.subs({E_m: E_matrix, B: B_matrix}).doit()
        K_matrix = t * K_matrix
        K_matrix = K_matrix.integrate((qsi, -1, 1), (eta, -1, 1))

        print("> K(e) Calculado")
        print("-----------------")

        matriz_K_está_pronta = True

    return np.array(K_matrix.evalf(subs={l: L, t: t_, v: v_, E: E_}))





def monitorar(mensagem, início):

    agora    = default_timer()
    até_aqui = agora - início

    global último_tempo
    na_operação  = até_aqui - último_tempo
    último_tempo = até_aqui

    print("> {: >10.5f}. {} ({:.5f})".format(até_aqui, mensagem, na_operação))
