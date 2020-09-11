import math

import sympy
import pytest

from suporte.elementos_finitos.membrana_quadrada import *


@pytest.fixture
def nós_de_teste():
    return [Nó(0, 1), Nó(1, 1), Nó(1, 0), Nó(0, 0), Nó(2, 1), Nó(2, 0)]


@pytest.fixture
def emqs_de_teste(nós_de_teste):
    return [MembranaQuadrada((nós_de_teste[0], nós_de_teste[1], nós_de_teste[2], nós_de_teste[3])),
            MembranaQuadrada((nós_de_teste[1], nós_de_teste[4], nós_de_teste[5], nós_de_teste[2]))]


def teste_intersecção_de_elementos_de_membrana_quadrada(emqs_de_teste, nós_de_teste):
    emq1, emq2 = emqs_de_teste
    emq1.traçar_bordas()
    emq2.traçar_bordas()

    assert set(emq1.bordas) & set(emq2.bordas) == {frozenset({nós_de_teste[1], nós_de_teste[2]})}


def teste_representação_de_elementos(emqs_de_teste):
    emq, _ = emqs_de_teste

    representação = "( 0.00,  1.00) - ( 1.00,  1.00)\n" \
                    "|                             |\n" \
                    "( 0.00,  0.00) - ( 1.00,  0.00)"

    assert str(emq) == representação


def teste_calcular():
    Ke = K_base.calcular({"l": 1, "t": 1, "v": 0.5, "E": 1})

    l = 1
    t = 1
    v = 0.5
    E = 1

    # Resultado teórico
    assert math.isclose(Ke[0, 0], (2 * t * E * (v - 3)) / (3 * (l ** 2) * (v ** 2 - 1)))


@pytest.mark.skip(reason="Computacionalmente custoso.")
def teste_construir(monkeypatch):
    K_matriz, dicionário_de_símbolos = K_base.construir()

    assert K_matriz == K_base.matriz
