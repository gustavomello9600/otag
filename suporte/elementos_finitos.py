import pickle
from math import isclose
from pathlib import Path
from timeit import default_timer

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

from suporte.membrana_quadrada import elemento_de_membrana_quadrada


class Nó:

    def __init__(self, coordenadas):
        self.x, self.y = coordenadas

    def __gt__(self, other):
        if self.y > other.y:
            return True
        elif self.y == other.y:
            return self.x < other.x
        else:
            return False

    def __lt__(self, other):
        return not (self == other or self > other)

    def __eq__(self, other):
        tol = 1e-10
        return (isclose(self.x, other.x, rel_tol=tol) and isclose(self.y, other.y, rel_tol=tol))

    def __str__(self):
        return ("Nó({}, {})".format(self.x, self.y))

    def __repr__(self):
        return ("Nó({}, {})".format(self.x, self.y))

    def def_ind(self, índice):
        self.índice = índice
        return self


class Elemento:

    def __init__(self, nós):
        self.nós = tuple(nós)
        self.lado = nós[1].x - nós[0].x
        self.traçar_bordas()

    def traçar_bordas(self):
        """Cria um atributo de bordas como uma tupla de duplas de Nós que compartilham lados"""
        self.bordas = tuple((nó, self.nós[i + 1 if i < 3 else 0]) for i, nó in enumerate(self.nós))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        borda_superior = "{} — {}".format(self.nós[0], self.nós[1])
        borda_inferior = "{} — {}".format(self.nós[2], self.nós[3])
        meio = "|" + (max([len(borda_superior), len(borda_inferior)]) - 2) * " " + "|"
        return "\n".join([borda_superior, meio, borda_inferior])


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
            i0, i1 = self.índice_do_nó(lado[0]), self.índice_do_nó(lado[1])
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

    def índice_do_nó(self, nó):
        return self.nós.index(nó)


def monitorar(mensagem="Não definido"):
    def auxiliar_do_monitorador(método):
        def monitorador(self, *args, **kwargs):
            if self._monitoramento_ativo:
                antes = default_timer()

            retorno = método(self, *args, **kwargs)

            if self._monitoramento_ativo:
                agora          = default_timer()
                no_método      = agora - antes
                desde_o_início = agora - self._início_do_monitoramento

                print(f"> {desde_o_início: >10.5f}. {mensagem} ({no_método:.5f})")

            return retorno
        return monitorador
    return auxiliar_do_monitorador()


class Resolvedor:

    def __init__(self, parâmetros_do_problema, método_padrão=None):
        self.parâmetros_do_problema = parâmetros_do_problema
        self._método_padrão = método_padrão

        self._monitoramento_ativo = False
        self._início_do_monitoramento = None

    def resolva_para(self, parâmetros_dos_elementos, malha, método=None, monitorar=False):
        self.configurar_monitoramento(monitorar)

        graus_de_liberdade = self.determinar_graus_de_liberdade()

        Ks_locais = self.calcular_matrizes_de_rigidez_local(**parâmetros_dos_elementos)
        K = self.montar_matriz_de_rigidez_geral(malha, Ks_locais, graus_de_liberdade,
                                                método=método if método is not None else self._método_padrão)

        f, u, ifc, iuc = self.incorporar_condições_de_contorno(self.parâmetros_do_problema,
                                                               parâmetros_dos_elementos,
                                                               graus_de_liberdade,
                                                               malha)

        # Lógica de solução do sistema linear
        Kfc    = self.onde_f_é_conhecido_fatiar(K, ifc)
        ufc    = self.resolver_sistema(Kfc, f[ifc])
        u[ifc] = ufc
        Kuc    = K[np.ix_(iuc, iuc)]
        f[iuc] = Kuc @ u[iuc]

        self.desligar_monitoramento()

        return f, u, malha

    def configurar_monitoramento(self, monitorar):
        if monitorar:
            self._monitoramento_ativo = True
            self._última_medição = self._início_do_monitoramento = default_timer()

    def desligar_monitoramento(self):
        self._monitoramento_ativo = False
        self._última_medição = self._início_do_monitoramento = None

    def determinar_graus_de_liberdade(self):
        pass

    @monitorar(mensagem="Matrizes de Rigidez Local Calculadas")
    def calcular_matrizes_de_rigidez_local(self, **parâmetros_dos_elementos):
        pass

    @monitorar(mensagem="Matrizes de Rigidez Global Acoplada")
    def montar_matriz_de_rigidez_geral(self, Ks_locais, graus_de_liberdade, malha, método):
        pass

    @monitorar(mensagem="Condições de Contorno Incorporadas")
    def incorporar_condições_de_contorno(self, parâmetros_do_problema, parâmetros_dos_elementos,
                                         graus_de_liberdade, malha):
        pass

    @monitorar(mensagem="K fatiado onde f é conhecido")
    def onde_f_é_conhecido_fatiar(self, K, ifc):
        return K[np.ix_(ifc, ifc)]

    @monitorar(mensagem="Sistema linear resolvido onde f é conhecido")
    def resolver_sistema(self, Kfc, f_ifc):
        return solve(Kfc, f_ifc)


def resolva_para(n=2, P=100e6, malha=None, padrão=True, método="OptV2"):
    """Retorna u e f para uma malha de ordem n e carga aplicada P"""

    lq   = 1/n
    gsdl = 2*len(malha.nós)

    K_emq = conseguir_matriz_de_rigidez_local(lq, padrão)
    K     = montar_matriz_de_rigidez_geral(K_emq, gsdl, malha, método)

    f, u, ifc, iuc = incorporar_condições_de_contorno(P, gsdl, malha, n)

    Kfc = K[np.ix_(ifc, ifc)]

    ufc = solve(Kfc, f[ifc])
    u[ifc] = ufc

    Kuc = K[np.ix_(iuc, iuc)]

    f[iuc] = Kuc @ u[iuc]

    return f, u, malha


def conseguir_matriz_de_rigidez_local(lq, padrão):
    if padrão:
        with open(Path(__file__).parent / "K_emq.b", "rb") as arquivo_de_cache:
            K_emq = pickle.load(arquivo_de_cache)
    else:
        with open(Path(__file__).parent / "K_emq.b", "wb") as arquivo_de_cache:
            K_emq = elemento_de_membrana_quadrada(lq)
            pickle.dump(arquivo_de_cache)
    return K_emq

def montar_matriz_de_rigidez_geral(K_emq, gsdl, malha, método):
    K = None
    if método == "expansão":
        Kes = dict()
        for elemento in malha.elementos:
            Ke = np.zeros((gsdl, gsdl))

            índices = np.array([
                [2 * malha.índice_do_nó(n), 2 * malha.índice_do_nó(n) + 1] for n in elemento.nós
            ]).flatten()

            for ie in range(len(índices)):
                for je in range(len(índices)):
                    i = índices[ie]
                    j = índices[je]

                    Ke[i][j] = K_emq[ie][je]

            Kes[elemento] = Ke

        K = sum(Kes.values())

    elif método == "compacto":
        K = np.zeros((gsdl, gsdl))

        índices = dict()
        for elemento in malha.elementos:
            índices[elemento] = np.array([[2 * malha.índice_do_nó(n), 2 * malha.índice_do_nó(n) + 1]
                                          for n in elemento.nós]).flatten()

        for i in range(8):
            for j in range(8):
                for e in malha.elementos:
                    p = índices[e][i]
                    q = índices[e][j]

                    K[p][q] += K_emq[i][j]

    elif método == "OptV1":
        from scipy.sparse import csr_matrix

        D = np.zeros(64 * malha.ne)
        I = np.zeros(64 * malha.ne, dtype="int32")
        J = np.zeros(64 * malha.ne, dtype="int32")
        d = 0
        for e in range(malha.ne):
            for i in range(8):
                for j in range(8):
                    I[d] = malha.me[i][e]
                    J[d] = malha.me[j][e]
                    D[d] = K_emq[i][j]
                    d += 1
        K = csr_matrix((D, (I, J)), shape=(gsdl, gsdl)).toarray()

    elif método == "OptV2":
        from scipy.sparse import csr_matrix
        ne = malha.ne

        D = np.zeros((64, ne))
        I = np.zeros((64, ne), dtype="int32")
        J = np.zeros((64, ne), dtype="int32")

        d = 0
        for i in range(8):
            for j in range(8):
                D[d, :] = np.repeat(K_emq[i][j], ne)
                I[d, :] = malha.me[i, :]
                J[d, :] = malha.me[j, :]
                d += 1
        K = csr_matrix((D.flat, (I.flat, J.flat)), shape=(gsdl, gsdl)).toarray()

    else:
        raise NotImplementedError
    return K


def incorporar_condições_de_contorno(P, gsdl, malha, n):

    f = np.zeros(gsdl)
    u = np.zeros(gsdl)
    u[:] = np.nan

    # Condições de Contorno em u
    for i in range(n + 1):
        try:
            i1 = 2 * malha.nós.index(Nó((0, 1 - i / n)))
        except ValueError:
            continue
        i2 = i1 + 1
        u[i1:(i2 + 1)] = 0
        f[i1:(i2 + 1)] = np.nan

    # Condições de Contorno em f
    gdl_P = grau_de_liberdade_associado_a_P = malha.nós.index(Nó((2, 0.5))) * 2 + 1
    f[gdl_P] = -P

    ifc = índices_onde_f_é_conhecido = np.where(~np.isnan(f))[0]
    iuc = índices_onde_u_é_conhecido = np.where(~np.isnan(u))[0]

    return f, u, ifc, iuc