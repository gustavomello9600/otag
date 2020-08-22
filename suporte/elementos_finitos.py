from typing import Any
from math import isclose
from timeit import default_timer
from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import solve
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


class Problema:

    def __init__(self, parâmetros_do_problema, método_padrão=None):
        self.parâmetros_do_problema = parâmetros_do_problema
        self._método_padrão = método_padrão

        self._monitoramento_ativo = False
        self._início_do_monitoramento = None

    class Monitorador:

        def __init__(self, mensagem="Não definido"):
            self._mensagem = mensagem

        def __call__(self, método):
            def monitorador(resolvedor, *args, **kwargs):
                if resolvedor._monitoramento_ativo:
                    antes = default_timer()

                    retorno = método(resolvedor, *args, **kwargs)

                    agora = default_timer()
                    no_método = agora - antes
                    desde_o_início = agora - resolvedor._início_do_monitoramento

                    print(f"> {desde_o_início: >10.5f}. {self._mensagem} ({no_método:.5f})")

                    return retorno
                else:
                    return método(resolvedor, *args, **kwargs)

            return monitorador

    def resolver_para(self, parâmetros_dos_elementos, malha, método=None, monitorar=False):
        self.configurar_monitoramento(monitorar)

        graus_de_liberdade = self.determinar_graus_de_liberdade(malha)

        Ks_locais = self.calcular_matrizes_de_rigidez_local(**parâmetros_dos_elementos)
        K = self.montar_matriz_de_rigidez_geral(malha, Ks_locais, graus_de_liberdade,
                                                método=método if método is not None else self._método_padrão)

        f, u, ifc, iuc = self.incorporar_condições_de_contorno(malha,
                                                               graus_de_liberdade,
                                                               **self.parâmetros_do_problema)

        # Lógica de determinação de f e u
        Kfc = self.onde_f_é_conhecido_fatiar(K, ifc)
        ufc = self.resolver_sistema_linear(Kfc, f[ifc])

        self.atualizar_graus_de_liberdade(u, ifc, ufc)

        Kuc = self.onde_u_é_conhecido_fatiar(K, iuc)
        self.atualizar_valores_de(f, iuc, Kuc @ u[iuc])

        self.desligar_monitoramento()

        return f, u, malha

    def configurar_monitoramento(self, monitorar):
        if monitorar:
            self._monitoramento_ativo = True
            self._última_medição = self._início_do_monitoramento = default_timer()

    def desligar_monitoramento(self):
        self._monitoramento_ativo = False
        self._última_medição = self._início_do_monitoramento = None

    def determinar_graus_de_liberdade(self, malha):
        return 0

    @Monitorador(mensagem="Matrizes de rigidez local determinadas")
    def calcular_matrizes_de_rigidez_local(self, **parâmetros_dos_elementos):
        return list()

    @Monitorador(mensagem="Matriz de rigidez global montada")
    def montar_matriz_de_rigidez_geral(self, malha, Ks_locais, graus_de_liberdade, método):
        return np.empty((0, 0))

    @Monitorador(mensagem="Condições de contorno incorporadas")
    def incorporar_condições_de_contorno(self, malha, graus_de_liberdade, **parâmetros_do_problema):
        return np.empty(0), np.empty(0), list(), list()

    @Monitorador(mensagem="K fatiado onde f é conhecido")
    def onde_f_é_conhecido_fatiar(self, K, ifc):
        return K[np.ix_(ifc, ifc)]

    @Monitorador(mensagem="Sistema linear resolvido onde f é conhecido")
    def resolver_sistema_linear(self, Kfc, f_ifc):
        return solve(Kfc, f_ifc)

    @Monitorador(mensagem="Graus de liberdade atualizados com o resultado da etapa anterior")
    def atualizar_graus_de_liberdade(self, u, ifc, ufc):
        u[ifc] = ufc

    @Monitorador(mensagem="K fatiado onde u é conhecido")
    def onde_u_é_conhecido_fatiar(self, K, iuc):
        return K[np.ix_(iuc, iuc)]

    @Monitorador(mensagem="Valores de f atualizados pela multiplicação de K por u")
    def atualizar_valores_de(self, f, iuc, valores):
        f[iuc] = valores
