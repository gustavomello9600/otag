#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from copy import copy
from random import choice, seed

# Importa os objetos necessários para discretizar o problema 
from suporte.elementos.membrana_quadrada import Nó, Elemento, Malha

# Importa a função que retorna os vetores de deslocamento $\vec{u}$ e o campo 
# de forças $\vec{v}$ para uma dada malha, com carga P e refinamento de ordem n
from suporte.elementos.membrana_quadrada import resolva_para

# Importa o objeto base do algoritmo genético
from suporte.algoritmo_genético import Indivíduo, População


virar_bit = np.vectorize(lambda b: 1 - b)


class Projeto(Indivíduo):
    
    def gerar_id_do_gene(self):
        return self.gene.data.tobytes()

# Ensina ao Python como trabalhar com Populações de Projetos
class População_de_Projetos(População):

    genes_úteis_testados = dict()

    # Ensina a construir os primeiros Indivíduos (Projetos) da População
    def geração_0(self):
        projetos = []

        for k in range(100):
            gene = np.random.choice((True, False), (38, 76))

            I    = []
            J    = []

            i = 19
            j = 75
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
                elif i == 37:
                    direções.remove("baixo")

                mover_para = choice(direções)
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

            gene[I, J] = 1
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
        
        return Projeto(gene_novo, "G{}_{}".format(self.n_da_geração, índice))
    
    def mutação(self, g):
        Médias   = sum([ind.gene for ind in self.indivíduos])/self.n_de_indivíduos
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

        gene_útil = np.zeros((38, 76))

        l = 1/38

        i = 19
        j = 75
        y = 1 - i*l
        elementos_conectados = []
        nós  = []
        i_nó = 0
        índice_por_nó = dict()
        me  = np.empty((8, 0), dtype="int16")


        buscando = True
        possíveis_ramificações = []
        descida = True
        subida  = False
        último_movimento = "esquerda"
        borda_alcançada = False

        while buscando:

            # busca vertical
            partida = (i, j)

            # começar descida
            while descida:

                # consigo descer mais?
                if i != 37:
                    abaixo = ind.gene[i + 1][j]
                    descida = abaixo
                else:
                    descida = False

                # há ramificações possíveis aqui do lado?
                if j != 75 and último_movimento != "esquerda":
                    direita = ind.gene[i][j + 1]
                    if direita and not gene_útil[i][j + 1]:
                        possíveis_ramificações.append((i, j + 1, "direita"))

                if j == 0:
                    borda_alcançada = True

                if j != 0 and último_movimento != "direita":
                    esquerda = ind.gene[i][j - 1]
                    if esquerda and not gene_útil[i][j - 1]:
                        possíveis_ramificações.append((i, j - 1, "esquerda"))

                # Adiciona este elemento à malha

                gene_útil[i][j] = 1

                y = 1 - i * l
                ul, ur, dr, dl = Nó((j * l, y)), Nó(((j + 1) * l, y)), \
                                 Nó(((j + 1) * l, y - l)), Nó((j * l, y - l))

                índices = []
                for nó in [ul, ur, dr, dl]:
                    if nó not in nós:
                        nós.append(nó)
                        índice_por_nó[nó] = i_nó
                        i_nó += 1
                    índices.append(índice_por_nó[nó])

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

                elementos_conectados.append(Elemento([ul, ur, dr, dl]))

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
                        subida = ind.gene[partida[0] - 1][partida[1]]
                    else:
                        subida = False

                    if subida:
                        i = partida[0] - 1

            # começar subida
            while subida:

                # consigo subir mais?
                if i != 0:
                    acima = ind.gene[i - 1][j]
                    subida = acima
                else:
                    subida = False

                # há ramificações possíveis aqui do lado?
                if j != 75 and último_movimento != "esquerda":
                    direita = ind.gene[i][j + 1]
                    if direita and not gene_útil[i][j + 1]:
                        possíveis_ramificações.append((i, j + 1, "direita"))

                if j == 0:
                    borda_alcançada = True

                if j != 0 and último_movimento != "direita":
                    esquerda = ind.gene[i][j - 1]
                    if esquerda and not gene_útil[i][j - 1]:
                        possíveis_ramificações.append((i, j - 1, "esquerda"))

                # Adiciona este elemento à malha
                gene_útil[i][j] = 1

                y = 1 - i*l
                ul, ur, dr, dl = Nó((      j*l,     y)), Nó(((j + 1)*l,     y)), \
                                 Nó(((j + 1)*l, y - l)), Nó((      j*l, y - l))

                índices = []
                for nó in [ul, ur, dr, dl]:
                    if nó not in nós:
                        nós.append(nó)
                        índice_por_nó[nó] = i_nó
                        i_nó += 1
                    índices.append(índice_por_nó[nó])

                iul, iur, idr, idl = índices

                me = np.append(me, np.array([[    2*iul],
                                             [2*iul + 1],
                                             [    2*iur],
                                             [2*iur + 1],
                                             [    2*idr],
                                             [2*idr + 1],
                                             [    2*idl],
                                             [2*idl + 1]],
                                            dtype="int16"), axis=1)

                elementos_conectados.append(Elemento([ul, ur, dr, dl]))

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

        if not borda_alcançada:
            print("> Indivíduo {} desconectado da borda".format(ind.nome))
            ind.adaptação = 0

        else:
            if gene_útil.data.tobytes() not in self.genes_úteis_testados:
                timed = True if ind.nome.endswith("1") else False

                ind.f, ind.u, ind.malha = resolva_para(38, malha=Malha(elementos_conectados, nós, me))

                Acon = gene_útil.sum()*(l**2)
                Ades = ind.gene.sum()*(l**2) - Acon

                ind.adaptação = 1/(Acon + 0.1*Ades)

                print("> {} conectado à borda. Adaptação: {}".format(
                      ind.nome, ind.adaptação))
            else:
                ind.adaptação = self.genes_úteis_testados[gene_útil.data.tobytes()]
                print("> Adaptação de {} já era conhecida pelo seu gene útil".format(ind.nome))

if __name__ == "__main__":
    seed(0)
    np.random.seed(0)

    pop = População_de_Projetos()
    pop.próxima_geração()