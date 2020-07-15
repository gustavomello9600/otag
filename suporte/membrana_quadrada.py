#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np
from sympy import *
from math import isclose
from numpy.linalg import solve
from timeit import default_timer
from matplotlib import pyplot as plt

matriz_K_está_pronta = False
K_matrix             = None
l, t, v, E           = None, None, None, None

último_tempo = 0

def monitorar(mensagem, início):
    agora    = default_timer()
    até_aqui = agora - início

    global último_tempo
    na_operação  = até_aqui - último_tempo
    último_tempo = até_aqui

    print("> {: >10.5f}. {} ({:.5f})".format(até_aqui, mensagem, na_operação))

def elemento_de_membrana_quadrada(L, t_=0.01, v_=0.3, E_=210e9):
    global matriz_K_está_pronta
    global K_matrix
    global l, t, v, E
    
    if not matriz_K_está_pronta:
        print("-----------------")
        print("> Calculando K(e)")
        print("Necessário apenas uma vez")
        qsi, eta = symbols("xi eta")
    
        N1_expr = (1/4) * (1 - eta) * (1 - qsi)
        N2_expr = (1/4) * (1 + eta) * (1 - qsi)
        N3_expr = (1/4) * (1 + eta) * (1 + qsi)
        N4_expr = (1/4) * (1 - eta) * (1 + qsi)
    
        x, y, l, x1, y1 = symbols("x y l x_1 y_1")
    
        qsi_x = (2/l) * (x - x1 - l*(sqrt(2)/2))
        eta_y = (2/l) * (y - y1 - l*(sqrt(2)/2))
    
        N = dict()
        Ns = [0, N1_expr, N2_expr, N3_expr, N4_expr]
        for i in (x, y):
            for k in range(1, 4 + 1):
                N[(i, k)] = simplify((diff(Ns[k], qsi) * diff(qsi_x, i) + diff(Ns[k], eta) * diff(eta_y, i)))
    
        B_matrix = Matrix([[N[(x, (i//2) + 1)] if i % 2 == 0 else 0 for i in range(8)],
                           [N[(y, (i//2) + 1)] if i % 2 == 1 else 0 for i in range(8)],
                           [N[(y if i % 2 == 0 else x, (i//2) + 1)] for i in range(8)]])
    
        B = MatrixSymbol("B", 3, 8)
    
    
        t, v, E  = symbols("t nu E")
        E_m      = MatrixSymbol("E", 3, 3)
        E_matrix = simplify((E/(1 - v**2)) * Matrix([[1, v,        0],
                                                     [v, 1,        0],
                                                     [0, 0, (1 -v)/2]]))
        
        K_matrix = B.T * E_m * B
        K_matrix = K_matrix.subs({E_m: E_matrix, B: B_matrix}).doit()
        K_matrix = t * K_matrix
        K_matrix = K_matrix.integrate((qsi, -1, 1), (eta, -1, 1))
        
        print("> K(e) Calculado")
        print("-----------------")
        
        matriz_K_está_pronta = True
    
    return np.array(K_matrix.evalf(subs={l:L, t:t_, v:v_, E:E_}))


class Nó:

    def __init__(self, coordenadas):
        self.x, self.y = coordenadas
        
    def __gt__(self, other):
        if   self.y  > other.y:
            return True
        elif self.y == other.y:
            if self.x < other.x:
                return True
            else:
                return False
        else:
            return False
    
    def __lt__(self, other):
        if self == other or self > other: return False
        else: return True

    def __eq__(self, other):
        tol = 1e-10
        return (isclose(self.x, other.x, rel_tol=tol) and isclose(self.y, other.y, rel_tol=tol))
        
    def __str__(self):
        return ("Nó({}, {})".format(self.x, self.y))
    
    def __repr__(self):
        return ("Nó({}, {})".format(self.x, self.y))
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    
class Elemento:

    def __init__(self, nós, tipo="Membrana Quadrada"):
        self.nós  = tuple(sorted(nós, reverse=True))
        self.tipo = tipo
        
        if tipo == "Membrana Quadrada":
            if len(nós) != 4:
                raise ValueError("A membrana quadrada é formada por 4 nós. "
                                 "{} foram fornecidos".format(len(nós)))
            self.l = nós[1].x - nós[0].x
        
        self.traçar_bordas()
            
    def traçar_bordas(self):
        if self.tipo == "Membrana Quadrada":
            self.bordas = []
            ks = [1, 3, 0, 2]
            for i, nó in enumerate(self.nós):
                k = ks[i]
                self.bordas.append((nó, self.nós[k]))
            self.bordas = tuple(self.bordas)
            
    def __str__(self):
        return self.__repr__()
        
    def __repr__(self):
        borda_superior = "{} — {}".format(self.nós[0], self.nós[1])
        borda_inferior = "{} — {}".format(self.nós[2], self.nós[3])
        meio = "|" + (max([len(borda_superior), len(borda_inferior)]) - 2)*" " + "|"
        return "\n".join([borda_superior, meio, borda_inferior])
                
    
class Malha:

    def __init__(self, elementos, nós, me, ordenado=False):

        self.elementos = elementos
        self.ne        = len(elementos)
        self.nós       = nós
        self.me        = me

        if not ordenado:
            self.nós = sorted(self.nós, reverse=True)

        self.bordas_traçadas = False
    
    def plot(self, deslocamento=None, k=2):
        if not self.bordas_traçadas:
            self.traçar_bordas()

        plt.rcParams['figure.dpi'] = 200
        
        if deslocamento is None: deslocamento = np.zeros(2*len(self.nós))
        
        for lado in self.lados:            
            i0 ,  i1 = self.índice_do_nó(lado[0]), self.índice_do_nó(lado[1])
            dx0, dx1 =         deslocamento[2*i0],         deslocamento[2*i1]
            dy0, dy1 =     deslocamento[2*i0 + 1],     deslocamento[2*i1 + 1]
            
            X  ,   Y = [lado[0].x + k*dx0, lado[1].x + k*dx1], [lado[0].y + k*dy0, lado[1].y + k*dy1]
            if lado in self.bordas:
                plt.plot(X, Y, "k-")
            else:
                plt.plot(X, Y, "k--", lw = 0.2)
                
        plt.axvline(x=0, c="black", lw="3")
        plt.axes().set_aspect("equal")

        plt.show()

    def traçar_bordas(self):

        self.bordas    = []
        self.lados     = []

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
    
                
def criar_malha(n, timed=False, início=None, tipo="Placa em balanço 2 x 1"):
    l   = 1/n

    nós       = []
    elementos = []
    me        = np.zeros((      8, 2*(n**2)), dtype="int16" )

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

            elementos.append(Elemento([ul, ur, dr, dl]))
            ie, je   = e // (2*n), e % (2*n)
            me[:, e] = 2*np.array([[          ie*(2*n + 1) + je,   ie*(2*n + 1) + je + 1,
                                    (ie + 1)*(2*n + 1) + je + 1, (ie + 1)*(2*n + 1) + je]], dtype="int16"
                                  ).repeat(2) + np.array([0, 1, 0, 1, 0, 1, 0, 1])
            e += 1

    if timed: monitorar("Lista de nós e de elementos da malha criadas", início)

    nós = sorted(nós, reverse=True)
    if timed: monitorar("Lista de nós ordenada", início)

    return Malha(elementos, nós, me, ordenado=True)

def resolva_para(n=2, P=100e6, malha=None, padrão=True, timed=False, método="OptV2"):
    "Retorna u e f para uma malha de ordem n e carga aplicada P"
    início = None
    if timed: início = default_timer()
    if timed: monitorar("Começando monitoramento", início)

    if malha is None:
        malha = criar_malha(n, timed=timed, início=início)

    if timed: monitorar("Elemento construído", início)

    lq    = 1/n
    gsdl  = 2*len(malha.nós)

    if timed: monitorar("Parâmetros iniciais definidos", início)

    if padrão:
        K_emq = pickle.load(open(os.path.join(os.path.abspath(__file__), "..", "K_emq.b"), "rb"))
    else:
        K_emq = elemento_de_membrana_quadrada(lq)
        pickle.dump(K_emq, open(os.path.join(os.path.abspath(__file__), "..", "K_emq.b"), "wb"))

    if timed: monitorar("Matriz de rigidez de emq carregada", início)


    if método == "expansão":
        Kes = dict()
        for elemento in malha.elementos:
            Ke = np.zeros((gsdl, gsdl))

            índices = np.array([
                        [2*malha.índice_do_nó(n), 2*malha.índice_do_nó(n) + 1] for n in elemento.nós
                                ]).flatten()

            for ie in range(len(índices)):
                for je in range(len(índices)):
                    i = índices[ie]
                    j = índices[je]

                    Ke[i][j] = K_emq[ie][je]

            Kes[elemento] = Ke

        K = sum(Kes.values())

    elif método=="compacto":
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

        D  = np.zeros(64*malha.ne)
        I  = np.zeros(64*malha.ne, dtype="int32")
        J  = np.zeros(64*malha.ne, dtype="int32")
        d  = 0
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

    if timed: monitorar("Formação da matriz de rigidez geral", início)

    f    = np.zeros(gsdl)
    u    = np.zeros(gsdl)
    u[:] = np.nan

    if timed: monitorar("Vetores u e f iniciados", início)

    #Condições de Contorno em u
    for i in range(n + 1):
        try:
            i1 = 2 * malha.nós.index(Nó((0, 1 - i/n)))
        except KeyError:
            continue
        i2 = i1 + 1
        u[i1:(i2 + 1)] = 0
        f[i1:(i2 + 1)] = np.nan

    if timed: monitorar("Condições de Contorno em u aplicadas", início)

    #Condições de Contorno em f
    gdl_P = grau_de_liberdade_associado_a_P = malha.nós.index(Nó((2, 0.5)))*2 + 1
    f[gdl_P] = -P

    if timed: monitorar("Condições de Contorno em f aplicadas", início)

    ifc = índices_onde_f_é_conhecido = np.where(~np.isnan(f))[0]
    iuc = índices_onde_u_é_conhecido = np.where(~np.isnan(u))[0]

    if timed: monitorar("Delimitação das condições de contorno feitas", início)

    Kfc = K[np.ix_(ifc, ifc)]
    if timed: monitorar("Matriz K fatiada onde f é conhecida", início)

    ufc = solve(Kfc, f[ifc])
    if timed: monitorar("ufc obtido resolvendo sistema linear", início)
    u[ifc] = ufc

    f = K @ u
    if timed: monitorar("K multiplicada por u para atualizar f", início)

    return f, u, malha

if __name__ == "__main__":

    comparar = False

    def tempo_de_execução(f, *args, **kwargs):
        começo = default_timer()
        f(*args, **kwargs)
        return default_timer() - começo

    if comparar:
        X = range(2, 40, 2)
        métodos = ["OptV1", "OptV2"]
        Y_s = [[tempo_de_execução(resolva_para, n, método=método) for n in X]
               for método in métodos]

        for Y in Y_s:
            plt.plot(X, Y)
        plt.legend(métodos)
        plt.axhline(0.5, c="red", ls="--")

        plt.show()

    import sys

    sys.stdout = open("PerformanceOpenBLAS_me_e_npc_otimizado.txt", "w")
    print("Usando OpenBLAS\n----------------------------------------------------------")
    print("Malha 74x38\n")

    f, u, malha = resolva_para(38, timed=True)