from unittest.mock import Mock

import numpy as np
import matplotlib.pyplot as plt

from suporte.elementos_finitos.membrana_quadrada import *
from suporte.elementos_finitos import *
from visualizador.placa_em_balanço import plotar_malha


def criar_malha_cheia(n):
    l = 1/n

    nós       = []
    elementos = []
    me        = np.zeros((8, 2*(n**2)), dtype="int16" )

    e = 0
    for i in range(n, 0, -1):
        for j in range(2 * n):
            if i == n:
                if j == 0:
                    ul = Nó((      j*l,       i*l))
                    ur = Nó(((j + 1)*l,       i*l))
                    dr = Nó(((j + 1)*l, (i - 1)*l))
                    dl = Nó((      j*l, (i - 1)*l))
                    nós.extend([ul, ur, dr, dl])
                else:
                    ul = nós[-2 - (0 if j != 1 else 1)]
                    ur = Nó(((j + 1)*l,       i*l))
                    dr = Nó(((j + 1)*l, (i - 1)*l))
                    dl = nós[-1 - (0 if j != 1 else 1)]
                    nós.extend([ur, dr])

            else:
                if j == 0:
                    ul = nós[3 if i == n - 1 else (n - i)*(2*n + 1) + 1]
                    ur = nós[2 if i == n - 1 else (n - i)*(2*n + 1)]
                    dr = Nó(((j + 1)*l, (i - 1)*l))
                    dl = Nó((      j*l, (i - 1)*l))
                    nós.extend([dr, dl])
                else:
                    if i == n - 1:
                        ul = nós[2 if j == 1 else 2*j + 1]
                        ur = nós[2*(j+1) + 1]
                        dr = Nó(((j + 1) * l, (i - 1) * l))
                        dl = nós[-2 if j == 1 else -1]
                        nós.extend([dr])
                    else:
                        ul = nós[-2*(n + 1) if j != 1 else -2*(n + 1) - 1]
                        ur = nós[-2*(n + 1) + 1]
                        dr = Nó(((j + 1) * l, (i - 1) * l))
                        dl = nós[-1 if j != 1 else -2]
                        nós.extend([dr])

            elementos.append(MembranaQuadrada([ul, ur, dr, dl]))
            ie, je   = e // (2*n), e % (2*n)
            me[:, e] = 2*np.array([[          ie*(2*n + 1) + je,   ie*(2*n + 1) + je + 1,
                                    (ie + 1)*(2*n + 1) + je + 1, (ie + 1)*(2*n + 1) + je]], dtype="int16"
                                  ).repeat(2) + np.array([0, 1, 0, 1, 0, 1, 0, 1])
            e += 1

    nós = sorted(nós, reverse=True)

    return Malha(elementos, nós, me)


def teste_malha_cheia():
    assert len(criar_malha_cheia(38).elementos) == 38*75, f"O resultado deveria ser {38*76}"


def teste_traçar_bordas():
    ul = Nó(-1.0, 1.0)
    ur = Nó(1.0, 1.0)
    dr = Nó(1.0, -1.0)
    dl = Nó(-1.0, -1.0)

    elem = MembranaQuadrada([ul, ur, dr, dl])

    assert isinstance(elem.bordas[0], frozenset)
    print(elem.bordas)
    print(elem)


def teste_intersecção_de_elementos():

    nós = (Nó(0.5 * (x + 1), 0.5 * (-y + 2)) for y in range(3) for x in range(3))

    elem1 = MembranaQuadrada([nós[0], nós[1], nós[4], nós[3]])
    elem2 = MembranaQuadrada([nós[1], nós[2], nós[5], nós[4]])
    elem3 = MembranaQuadrada([nós[3], nós[4], nós[7], nós[6]])
    elem4 = MembranaQuadrada([nós[4], nós[5], nós[8], nós[7]])

    # Testa se a intersecção é encontrada
    assert (set(elem1.bordas) & set(elem2.bordas)).pop() == frozenset({nós[1], nós[4]})

    # Visualização da Malha
    me = np.array([[0,  2,  6,  8],
                   [1,  3,  7,  9],
                   [2,  4,  8, 10],
                   [3,  5,  9, 11],
                   [8, 10, 14, 16],
                   [9, 11, 15, 17],
                   [6,  8, 12, 14],
                   [7,  9, 13, 15]])

    malha_simples = Malha([elem1, elem2, elem3, elem4], nós, me)

    proj = Mock()
    proj.malha = malha_simples
    proj.u = 2*len(nós)*[0]

    fig, ax = plt.subplots()
    plotar_malha(proj, ax)
    plt.show()


def teste_buraco_na_malha():
    nós = [Nó(x/3 + 0.5, -y/3 + 1) for y in range(4) for x in range(4)]

    elem1 = MembranaQuadrada([nós[0], nós[1], nós[5], nós[4]])
    elem2 = MembranaQuadrada([nós[1], nós[2], nós[6], nós[5]])
    elem3 = MembranaQuadrada([nós[2], nós[3], nós[7], nós[6]])
    elem4 = MembranaQuadrada([nós[4], nós[5], nós[9], nós[8]])
    #elem5 é onde fica o vazio
    elem6 = MembranaQuadrada([nós[6], nós[7], nós[11], nós[10]])
    elem7 = MembranaQuadrada([nós[8], nós[9], nós[13], nós[12]])
    elem8 = MembranaQuadrada([nós[9], nós[10], nós[14], nós[13]])
    elem9 = MembranaQuadrada([nós[10], nós[11], nós[15], nós[14]])

    malha_simples = Malha([elem1, elem2, elem3, elem4, elem6, elem7, elem8, elem9], nós, None)

    proj = Mock()
    proj.malha = malha_simples
    proj.u = 2 * len(nós) * [0]

    fig, ax = plt.subplots()
    plotar_malha(proj, ax)
    plt.show()


if __name__ == "__main__":
    teste_buraco_na_malha()