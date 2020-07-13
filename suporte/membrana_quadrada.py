#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sympy import *
from copy import copy
from numpy.linalg import inv
from matplotlib import pyplot as plt

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
        
    def __eq__(self, other):
        (self.x, self.y) == (other.x, other.y)
    
    def __lt__(self, other):
        if self == other or self > other: return False
        else: return True
        
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

    def __init__(self, elementos):
        self.elementos = elementos
        self.bordas_de_elementos = [borda for bordas in [e.bordas for e in elementos]
                                          for borda in bordas]
        self.bordas = []
        self.lados  = []
        
        lista_teste = copy(self.bordas_de_elementos)
        while len(lista_teste) > 0:
            lado = lista_teste.pop(0)
            conjugado = (lado[1], lado[0])
            
            if conjugado not in lista_teste:
                self.bordas.append(lado)
            else:
                lista_teste.remove(conjugado)
            self.lados.append(lado)
            
        self.nós = sorted(list(set([nó for nós in self.lados for nó in nós])), reverse=True)
    
    def plot(self, deslocamento=None, k=2):
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
    
    def índice_do_nó(self, nó):
        return self.nós.index(nó)
    
                
def criar_malha(n, tipo="Placa em balanço 2 x 1"):
    l = 1/n
    nós = [[Nó((j*l, i*l)) for j in range(2*n + 1)]
                           for i in range(n, -1, -1)]
    
    elementos = []
    for i in range(n):
        for j in range(2*n):
            elementos.append(Elemento((nós[i][j], nós[i][j + 1], nós[i + 1][j + 1], nós[i + 1][j])))
    
    return Malha(elementos)

def resolva_para(n=2, P=100e6, malha=None):
    "Retorna u e f para uma malha de ordem n e carga aplicada P"

    if malha is None:
        malha = criar_malha(n)
        
    lq    = 1/n
    gsdl  = 2*len(malha.nós)
    Kes   = dict()
    
    K_emq = elemento_de_membrana_quadrada(lq)

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
    
    gdl_P = grau_de_liberdade_associado_a_P = (int((2*n + 1)*(n/2 + 1)) - 1)*2 + 1

    f = np.zeros(gsdl)
    u = np.zeros(gsdl)
    u[:] = np.nan

    #Condições de Contorno em u
    for i in range(n + 1):
        i1 = 2*i*(2*n + 1)
        i2 = i1 + 1
        u[i1:(i2 + 1)] = 0
        f[i1:(i2 + 1)] = np.nan

    #Condições de Contorno em f
    f[gdl_P] = -P

    ifc = índices_onde_f_é_conhecido = np.where(~np.isnan(f))[0]
    iuc = índices_onde_u_é_conhecido = np.where(~np.isnan(u))[0]

    Kfc = K[np.ix_(ifc, ifc)]
    ufc = inv(Kfc) @ f[ifc].T

    u[ifc] = ufc

    f = K @ u.T

    tolerância = 1e-5
    f[abs(f) < tolerância] = 0

    return f, u

if __name__ == "__main__":
    f, u = resolva_para(n=20)
    print(np.array_str(np.vstack([f, u]).T, precision=3))
    print("\n-----------------------------")
    print("> Resolvendo problema de novo")
    f, u = resolva_para()
    print("> Resolvido")