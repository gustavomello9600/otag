#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.cluster.hierarchy as sch

from copy import copy
from random import choice, seed, shuffle

from suporte.algoritmo_genético import Indivíduo, População
from suporte.membrana_quadrada import Nó, Elemento, Malha, resolva_para


class Projeto(Indivíduo):
    
    def gerar_id_do_gene(self):
        return self.gene.data.tobytes()

    def definir_espécie(self, espécie):
        self.espécie = espécie


class População_de_Projetos(População):

    alfa_0 = 10
    Dlim = 0.005
    genes_úteis_testados = dict()

    def __init__(self, indivíduos=None, pm=0.01/100):
        super(População_de_Projetos, self).__init__(indivíduos=indivíduos, pm=pm)
        self.perfis_das_espécies = dict()

    # Ensina a construir os primeiros Indivíduos (Projetos) da População
    def geração_0(self):
        projetos = []
        for k in range(100):
            grafo = np.random.choice((True, False), (7, 14))

            kis = fatiar_intervalo(c=38, t=4, f= 7, dividir_ao_meio=True)
            kjs = fatiar_intervalo(c=76, t=4, f=14, dividir_ao_meio=False)

            I = []
            J = []

            i = int(np.where(kis == 19)[0])
            j = 13
            alcançou_a_borda = False
            último_movimento = "direita"
            while not alcançou_a_borda:
                I.append(i)
                J.append(j)

                direções = ["cima", "baixo", "esquerda"]

                if último_movimento == "cima":
                    direções.remove("baixo")
                elif último_movimento == "baixo":
                    direções.remove("cima")

                if i == 0:
                    direções.remove("cima")
                elif i == 6:
                    direções.remove("baixo")

                if len(direções) == 1:
                    p = (1, )
                elif len(direções) == 2:
                    p = (0.64, 0.36)
                else:
                    p = (0.32, 0.32, 0.36)
                mover_para = np.random.choice(direções, p=p)
                if mover_para == "esquerda" and j == 0:
                    alcançou_a_borda = True
                else:
                    if mover_para == "esquerda":
                        j -= 1
                    elif mover_para == "cima":
                        i -= 1
                    else:
                        i += 1
                último_movimento = mover_para

            grafo[I, J] = 1

            gene = np.random.choice((True, False), (38, 76))

            for i in range(7):
                for j in range(14):
                    gene[kis[i]:kis[i + 1], kjs[j]:kjs[j + 1]] = grafo[i][j]

            projetos.append(Projeto(gene, "G0_{}".format(k + 1)))

        return projetos

    # Ensina como cruzar os genes dos Indivíduos (Projetos) selecionados para reprodução
    def crossover(self, p1, p2, índice):
        gene_novo = copy(p1.gene)
        
        i1, i2, j1, j2 = np.random.randint(1, 37), np.random.randint(1, 37),\
                         np.random.randint(1, 75), np.random.randint(1, 75)
        while i1 == i2:
            i2 = np.random.randint(1, 37)
        while j1 == j2:
            j2 = np.random.randint(1, 75)
            
        i_b, i_c = max([i1, i2]), min([i1, i2])
        j_d, j_e = max([j1, j2]), min([j1, j2])
        
        corte_horizontal = [(0, i_c), (i_c, i_b), (i_b, 76)]
        corte_vertical   = [(0, j_e), (j_e, j_d), (j_d, 38)]

        blocos = []

        while len(blocos) < 3:
            bloco = (choice(corte_horizontal), choice(corte_vertical))
            if bloco not in blocos:
                blocos.append(bloco)
                
        for bloco in blocos:
            ibc, ibb = bloco[0]
            jbe, jdd = bloco[1]
            
            gene_novo[ibc:ibb, jbe:jdd] = p2.gene[ibc:ibb, jbe:jdd]
        
        return Projeto(gene_novo, "G{}_{}".format(self.geração, índice))

    def próxima_geração(self):
        self.alfa = self.alfa_0 * (1.01 ** (self.geração))

        super(População_de_Projetos, self).próxima_geração()

    def mutação(self, g):
        Médias   = sum([ind.gene for ind in self.indivíduos])/self.n
        Médias_2 = Médias ** 2

        for ind in g:
            probabilidade_de_mutar = self.pm + 99*self.pm*Médias_2 + 99*self.pm*ind.gene*(1 - 2*Médias)
            mutações = probabilidade_de_mutar > np.random.random((38, 76))
            try:
                ind.gene[mutações] = virar_bit(ind.gene[mutações])
            except ValueError:
                pass

    def testar_adaptação(self, ind):
        ind.adaptação_testada = True

        l = 1/38

        gene_útil, borda_alcançada, elementos_conectados, nós, me = determinar_gene_útil(ind.gene, l)

        if not borda_alcançada:
            print("> Indivíduo {} desconectado da borda".format(ind.nome))
            ind.adaptação = 0

        else:
            if gene_útil.data.tobytes() not in self.genes_úteis_testados:
                timed = True if ind.nome.endswith("1") else False

                ind.f, ind.u, ind.malha = resolva_para(38,
                                                       P=100e3,
                                                       malha=Malha(elementos_conectados, nós, me),
                                                       timed=timed)

                Acon = gene_útil.sum() * (l ** 2)
                Ades = ind.gene.sum() * (l ** 2) - Acon

                n = len(nós)
                Dmax = np.sqrt(np.sum(ind.u.reshape((n, 2)) ** 2, axis=1).max())
                pena = 0 if Dmax < self.Dlim else Dmax - self.Dlim

                if pena:
                    print("> Indivíduo {} penalizado: Dmax - Dlim = {:.3e} metros".format(ind.nome, pena))

                ind.adaptação = 1 / (Acon + 0.4 * Ades + self.alfa * pena)

                print("> {} conectado à borda. Adaptação: {}".format(
                    ind.nome, ind.adaptação))
            else:
                ind.adaptação = self.genes_úteis_testados[gene_útil.data.tobytes()]
                print("> Adaptação de {} já era conhecida pelo seu gene útil".format(ind.nome))

    def seleção_natural(self):
        for ind in self.indivíduos:
            if not ind.adaptação_testada:
                self.conseguir_adaptação(ind)

        if self.perfis_das_espécies is None:
            indivíduos_selecionados = sorted(self.indivíduos, reverse=True)[:self.n//2]

            genes = np.array([ind.gene.flatten() for ind in indivíduos_selecionados])

            rede_de_conexões   = sch.linkage(genes, metric="correlation")
            espécies_dos_genes = sch.fcluster(rede_de_conexões, t=0.3, criterion="distance")
            espécies_iniciais  = único(espécies_dos_genes)

            self.perfis_das_espécies = []
            for espécie in espécies_iniciais:
                genes_da_espécie  = genes[espécies_dos_genes == espécie, :]
                perfil_da_espécie = np.mean(genes_da_espécie, axis=0)
                self.perfis_das_espécies[espécie] = perfil_da_espécie

            for i in range(len(indivíduos_selecionados)):
                indivíduos_selecionados[i].definir_espécie(espécies_dos_genes[i])

        indivíduos_selecionados = sorted(self.indivíduos, reverse=True)[:self.n // 2]

        return indivíduos_selecionados


# Bloco de Funções Auxiliares
# ---------------------------

virar_bit = np.vectorize(lambda b: 1 - b)

def único(X):
    """Retorna os valores únicos da lista X de forma ordenada"""
    _, índice = np.unique(X, return_index=True)
    return X[np.sort(índice)]

def fatiar_intervalo(c=38, t=4, f=7, dividir_ao_meio=False):
    """
    Fatia aleatoriamente um intervalo de comprimento c em f fatias (ou subintervalos) de comprimento mínimo t
    e retorna uma lista com os índices correspondentes aos pontos de corte.

    Argumentos
    ----------
    c              : int  = Comprimento do intervalo original
    t              : int  = Espessura mínima
    f              : int  = Número de fatias
    dividir_ao_meio: bool = Determina se é obrigatório dividir o intervalo de comprimento c na posição c // 2

    Retorna
    -------
    ks             : list = Lista de índices dos pontos de corte
    """

    # Calcula a folga do intervalo para a divisão esperada
    folga = c - f*t
    if folga <= 0:
        raise ValueError("Impossível dividir intervalo de comprimento {}"
                         " em {} fatias com {} de comprimento mínimo".format(c, f, t))

    # Calcula os índices como o resultado da soma cumulativa do vetor que contém o comprimento de cada
    # subintervalo tomado como o comprimento mínimo somado a uma distribuição aleatória da folga
    ks = np.cumsum(np.array(f*[t]) + np.array(distribuir_no_intervalo(folga, f)))

    # Corrige a divisão quando se deseja que haja um corte em c // 2
    if dividir_ao_meio:
        for i, k in enumerate(ks):
            if k >= c/2:
                if k == c/2:
                    break
                ks[i - 1] = c / 2
                j = i
                while ks[j] - ks[j - 1] < t:
                    ks[j] += t - (ks[j] - ks[j - 1])
                    if j == len(ks) - 1:
                        j = 1
                    else:
                        j += 1
                break

    return ks

def distribuir_no_intervalo(folga, fatias):
    """
    Distribui a folga aleatoriamente dentre as fatias

    Argumentos
    ----------
    folga       : int  = Tamanho da folga
    fatias      : int  = Número de fatias

    Retorna
    -------
    distribuição: list = Lista cujas posições contém a porção de folga que o intervalo correspondente recebeu
    """

    distribuição = []
    for _ in range(fatias):
        espaço_extra = choice(list(range(folga + 1)))
        folga -= espaço_extra
        distribuição.append(espaço_extra)

    if folga > 0:
        distribuição[-1] += folga

    shuffle(distribuição)
    return distribuição

def determinar_gene_útil(gene, l):
    """
    Executa um algoritmo de busca responsável por determinar, para um certo gene cuja expressão fenotípica é dada
    por uma malha de elementos quadrados de lado l, a maior porção contínua de matéria satisfazendo as restrições do
    problema, isto é, estar conectada simultaneamente ao ponto de aplicação da força e à borda.

    Também cuida de inicializar a malha correspondente à expressão fenotípica do gene e seus respectivos elementos e
    nós. Embora ter uma função que lide com tantas operações ao mesmo tempo não seja o padrão de programação
    recomendável na maioria dos casos, aqui se justifica pelo ganho em performance.

    Argumentos
    ----------
    gene           : np.array((38, 76)) = Matriz binária que carrega o código genético
    l              : int                = Comprimento do lado do elemento de membrana quadrado

    Retorna
    -------
    gene_útil      : np.array((38, 76)) = Matriz binária que carrega a porção do gene que forma o fenótipo
    borda_alcançada: bool               = O fenótipo se estende desde o ponto de aplicação da carga até a borda?
    elementos      : list               = Lista de Elementos que compõem a malha
    nós            : list               = Lista de Nós que compõem a malha
    me             : np.array(( 8, ne)) = Matriz de correspondência entre os índices locais e globais de cada grau
                                           de liberdade (gsdl = graus de liberdade)
    """
    gene_útil = np.zeros((38, 76))

    # Define a posição inicial do algoritmo de busca
    i = 19
    j = 75

    elementos = []
    nós       = []
    me        = np.empty((8, 0), dtype="int16")

    # Inicializa listas auxiliares que ajudam a manter curso dos índices dos nós
    nós_índices = []
    i_nó_na_malha = 0
    índice_na_malha = dict()

    # Inicializa parâmetros do algoritmo de busca
    buscando = True
    possíveis_ramificações = []
    descida = True
    subida = False
    último_movimento = "esquerda"
    borda_alcançada = False

    # Executa o algoritmo de busca
    while buscando:

        # busca vertical
        partida = (i, j)

        # começar descida
        while descida:

            # consigo descer mais?
            if i != 37:
                abaixo = gene[i + 1][j]
                descida = abaixo
            else:
                descida = False

            # há ramificações possíveis aqui do lado?
            if j != 75 and último_movimento != "esquerda":
                direita = gene[i][j + 1]
                if direita and not gene_útil[i][j + 1]:
                    possíveis_ramificações.append((i, j + 1, "direita"))

            if j == 0:
                borda_alcançada = True

            if j != 0 and último_movimento != "direita":
                esquerda = gene[i][j - 1]
                if esquerda and not gene_útil[i][j - 1]:
                    possíveis_ramificações.append((i, j - 1, "esquerda"))

            # Adiciona este elemento à malha

            gene_útil[i][j] = 1

            y = 1 - i * l
            ul, ur, dr, dl = Nó((j * l, y)).def_ind((i, j)), \
                             Nó(((j + 1) * l, y)).def_ind((i, j + 1)), \
                             Nó(((j + 1) * l, y - l)).def_ind((i + 1, j + 1)), \
                             Nó((j * l, y - l)).def_ind((i + 1, j))

            índices = []
            for nó in [ul, ur, dr, dl]:
                if nó.índice not in nós_índices:
                    nós.append(nó)
                    nós_índices.append(nó.índice)
                    índice_na_malha[nó.índice] = i_nó_na_malha
                    i_nó_na_malha += 1
                índices.append(índice_na_malha[nó.índice])

            iul, iur, idr, idl = índices

            me = np.append(me, np.array([[2 * iul],
                                         [2 * iul + 1],
                                         [2 * iur],
                                         [2 * iur + 1],
                                         [2 * idr],
                                         [2 * idr + 1],
                                         [2 * idl],
                                         [2 * idl + 1]],
                                        dtype="int16"), axis=1)

            elementos.append(Elemento([ul, ur, dr, dl]))

            # Remove a coordenada das possíveis novas ramificações
            try:
                possíveis_ramificações.remove((i, j, "esquerda"))
            except ValueError:
                try:
                    possíveis_ramificações.remove((i, j, "direita"))
                except ValueError:
                    pass

            # Decide se continua descendo ou se passa a subir
            if descida:
                i = i + 1
                último_movimento = "baixo"
            else:
                if partida[0] != 0:
                    subida = gene[partida[0] - 1][partida[1]]
                else:
                    subida = False

                if subida:
                    i = partida[0] - 1

        # começar subida
        while subida:

            # consigo subir mais?
            if i != 0:
                acima = gene[i - 1][j]
                subida = acima
            else:
                subida = False

            # há ramificações possíveis aqui do lado?
            if j != 75 and último_movimento != "esquerda":
                direita = gene[i][j + 1]
                if direita and not gene_útil[i][j + 1]:
                    possíveis_ramificações.append((i, j + 1, "direita"))

            if j == 0:
                borda_alcançada = True

            if j != 0 and último_movimento != "direita":
                esquerda = gene[i][j - 1]
                if esquerda and not gene_útil[i][j - 1]:
                    possíveis_ramificações.append((i, j - 1, "esquerda"))

            # Adiciona este elemento à malha
            gene_útil[i][j] = 1

            y = 1 - i * l
            ul, ur, dr, dl = Nó((j * l, y)).def_ind((i, j)), \
                             Nó(((j + 1) * l, y)).def_ind((i, j + 1)), \
                             Nó(((j + 1) * l, y - l)).def_ind((i + 1, j + 1)), \
                             Nó((j * l, y - l)).def_ind((i + 1, j))

            índices = []
            for nó in [ul, ur, dr, dl]:
                if nó.índice not in nós_índices:
                    nós.append(nó)
                    nós_índices.append(nó.índice)
                    índice_na_malha[nó.índice] = i_nó_na_malha
                    i_nó_na_malha += 1
                índices.append(índice_na_malha[nó.índice])

            iul, iur, idr, idl = índices

            me = np.append(me, np.array([[2 * iul],
                                         [2 * iul + 1],
                                         [2 * iur],
                                         [2 * iur + 1],
                                         [2 * idr],
                                         [2 * idr + 1],
                                         [2 * idl],
                                         [2 * idl + 1]],
                                        dtype="int16"), axis=1)

            elementos.append(Elemento([ul, ur, dr, dl]))

            # Remove a coordenada das possíveis novas ramificações
            try:
                possíveis_ramificações.remove((i, j, "esquerda"))
            except ValueError:
                try:
                    possíveis_ramificações.remove((i, j, "direita"))
                except ValueError:
                    pass

            # Decide se continua descendo ou se passa a subir
            if subida:
                i = i - 1
                último_movimento = "cima"

        if len(possíveis_ramificações) > 0:
            i, j, último_movimento = possíveis_ramificações.pop(-1)
            descida = True
            subida = False

        else:
            buscando = False

    return gene_útil, borda_alcançada, elementos, nós, me
