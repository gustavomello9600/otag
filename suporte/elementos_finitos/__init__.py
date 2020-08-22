from typing import Any
from math import isclose
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Nó:

    x: float = field(hash=False)
    y: float = field(hash=False)
    etiqueta: Any = field(default=None, hash=True, compare=False)

    def __gt__(self, other):
        if self.y > other.y:
            return True
        elif self.y == other.y:
            return self.x < other.x
        else:
            return False

    def __lt__(self, other):
        if self.y < other.y:
            return True
        elif self.y == other.y:
            return self.x > other.x
        else:
            return False

    def __eq__(self, other):
        tol = 1e-10
        return (isclose(self.x, other.x, rel_tol=tol) and isclose(self.y, other.y, rel_tol=tol))


class Elemento:

    def __init__(self, nós):
        self.nós = tuple(nós)
        self.traçar_bordas()

    def __str__(self):
        return self.__repr__()

    # Sobrescrever nas classes filhas
    def traçar_bordas(self):
        pass


class Malha:

    def __init__(self, elementos, nós, me):

        self.elementos = elementos
        self.ne = len(elementos)
        self.nós = nós
        self.me = me

        self.bordas_traçadas = False

    def plot(self, deslocamento=None, k=1, show=True):
        if not self.bordas_traçadas:
            self.traçar_bordas()

        plt.rcParams['figure.dpi'] = 200
        fig, ax = plt.subplots()

        if deslocamento is None:
            deslocamento = np.zeros(2 * len(self.nós))

        for lado in self.lados:
            i0, i1 = self.índice_de(lado[0]), self.índice_de(lado[1])
            dx0, dx1 = deslocamento[2 * i0], deslocamento[2 * i1]
            dy0, dy1 = deslocamento[2 * i0 + 1], deslocamento[2 * i1 + 1]

            X, Y = [lado[0].x + k * dx0, lado[1].x + k * dx1], [lado[0].y + k * dy0, lado[1].y + k * dy1]
            if lado in self.bordas:
                plt.plot(X, Y, "k-")
            else:
                plt.plot(X, Y, "k--", lw=0.2)

        plt.axvline(x=0, c="black", lw="3")
        plt.xlim((-0.5, 2.5))
        plt.ylim((-0.5, 1.5))
        ax.set_aspect('equal')

        if show:
            plt.show()

    def traçar_bordas(self):

        self.bordas = []
        self.lados = []

        for elemento in self.elementos:
            for lado in elemento.bordas:
                if (lado[1], lado[0]) in self.bordas:
                    self.lados.append(lado)
                    self.bordas.remove((lado[1], lado[0]))
                else:
                    self.bordas.append(lado)
        self.lados.extend(self.bordas)

        self.bordas_traçadas = True

    def índice_de(self, nó):
        return self.nós.index(nó)