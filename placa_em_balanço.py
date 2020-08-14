#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.cluster.hierarchy as sch

from copy import copy
from random import choice, shuffle

from suporte.algoritmo_genético import Indivíduo, População
from suporte.membrana_quadrada import Nó, Elemento, Malha, resolva_para


class Projeto(Indivíduo):
    """
    Classe que carrega as propriedades de cada projeto. Herda seus principais atributos e métodos da classe Indivíduo
    do módulo algoritmo_genético.py

    Métodos Sobrescritos
    --------------------
    gerar_id_do_gene(self: Projeto) -> bytes
        Gera uma representação em bytestring do gene. Útil para comparar dois indivíduos.

    Propriedades
    ------------
    espécie: int -- Número da espécie a qual o indivíduo pertence
    """

    def __init__(self, *args, **kwargs):
        super(Projeto, self).__init__(*args, **kwargs)
        self._espécie = None

    def gerar_id_do_gene(self):
        return self.gene.data.tobytes()

    @property
    def espécie(self):
        return self._espécie

    @espécie.setter
    def espécie(self, espécie):
        self._espécie = espécie


class PopulaçãoDeProjetos(População):
    """
    Classe de objetos que agregam os indivíduos de uma população de projetos e os manipulam de acordo
    com operadores genéticos específicos ao problema. Herda seus atributos da classe População definida em
    algoritmo_genético.py e tem a maior parte dos seus métodos sobrescritos aqui.

    Atributos (da classe)
    ---------------------
    alfa_0              : Number -- Valor inicial do coeficiente que penaliza o excesso de deslocamento.
    Dlim                : float  -- Deslocamento limite no problema em análise.
    genes_úteis_testados: dict   -- Dicionário que armazena e recupera as adaptações dos genes já testados.

    Atributos (do objeto)
    ---------------------
    perfis_das_espécies : dict   -- Para cada identificador de espécie, armazena e recupera seu perfil.
    alfa                : Number -- Valor corrente do coeficiente alfa atualizado a cada geração

    Métodos
    -------
    OPERADORES GENÉTICOS
    próxima_geração() -> None
        Executa todos os passos necessários para avançar uma geração como implementado pela classe População.
        Adicionalmente, atualiza o valor de alfa.
    geração_0(t: int) -> projetos: List[Projeto]
        Gera aleatoriamente 100 projetos de espessura interna mínima igual a t que estão conectados à borda
        e ao ponto de aplicação da carga.

    ESTÁTICOS (AUXILIARES)
    fatiar_intervalo(c: int, t: int, f: int, dividir_ao_meio: bool) -> ks: List[int]
        Fatia aleatoriamente um intervalo de comprimento c em f fatias (ou subintervalos) de comprimento mínimo t
        e retorna uma lista com os índices correspondentes aos pontos de corte.
    distribuir(folga: int, fatias: int) -> distribuição: List[int]
        Distribui uma folga aleatoriamente dentre um dado número de fatias.
    caminhar_até_a_borda(i_partida: int, j_partida: int) -> I: List[int], J: List[int]
        Caminha aleatoriamente por um grafo desde o ponto de aplicação da força até uma borda.
    formar_a_partir_do(grafo: np.array((7, 14), kis: List[int], kjs: List[int]) -> gene: np.array((38, 76))
        Recupera a informação do grafo de partição do espaço de projeto para formar o gene.
    """

    alfa_0 = 10
    Dlim   = 0.005
    genes_úteis_testados = dict()

    def __init__(self, indivíduos=None, pm=0.01 / 100):
        super(PopulaçãoDeProjetos, self).__init__(indivíduos=indivíduos, pm=pm)
        self.perfis_das_espécies = dict()
        self.alfa = self.alfa_0

    def próxima_geração(self):
        self.alfa = self.alfa_0 * (1.01 ** (self.geração))
        super(PopulaçãoDeProjetos, self).próxima_geração()

    def geração_0(self, t=4):
        """
        Gera aleatoriamente 100 projetos de espessura interna mínima igual a t que estão conectados à borda
        e ao ponto de aplicação da carga.

        Fatia o espaço de projeto em 7 x 14 pedaços interpretados como nós de um grafo. Inicialmente, atribui
        um valor binário qualquer a cada nó e em seguida executa um algoritmo simples de caminhada aleatória desde
        o ponto de aplicação da carga até a borda. Os nós pertencentes à trajetória percorrida no grafo recebem o valor
        1. O grafo é então traduzido para um gene que inicia uma instância da classe Projeto. Este processo é repetido
        e o resultado de 100 iterações é retornado numa lista.

        Argumentos
        ----------
        t       : int            -- Espessura interna mínima

        Retorna
        -------
        projetos: List[Projetos] -- Instâncias da classe Projeto que carregam genes adequados ao problema
        """

        projetos = []
        for k in range(100):
            # Inicia um grafo que representa o preenchimento de cada fatia do espaço de projeto
            grafo = np.random.choice((True, False), (7, 14))

            # Determina os pontos de corte e retorna seus índices i e j
            kis = self.fatiar_intervalo(c=38, t=t, f=7, dividir_ao_meio=True)
            kjs = self.fatiar_intervalo(c=76, t=t, f=14, dividir_ao_meio=False)

            # Caminha aleatoriamente pelo grafo desde o ponto de aplicação da força até uma borda,
            # preenchendo as fatias ao longo da trajetória
            trajetória = self.caminhar_até_a_borda(i_partida=int(np.where(kis == 19)[0]), j_partida=13)
            grafo[trajetória[0], trajetória[1]] = 1

            gene = self.formar_a_partir_do(grafo, kis, kjs)

            # Inicializa uma instância de Projeto a partir do gene, atribuindo-lhe um nome G0_x que representa
            # o X-ésimo indivíduo da geração 0, e a põe na lista de Projetos
            projetos.append(Projeto(gene, "G0_{}".format(k + 1)))

        return projetos

    # Método Auxiliar
    def fatiar_intervalo(self, c=38, t=4, f=7, dividir_ao_meio=False):
        """
        Fatia aleatoriamente um intervalo de comprimento c em f fatias (ou subintervalos) de comprimento mínimo t
        e retorna uma lista com os índices correspondentes aos pontos de corte.

        Argumentos
        ----------
        c              : int  -- Comprimento do intervalo original
        t              : int  -- Espessura mínima
        f              : int  -- Número de fatias
        dividir_ao_meio: bool -- Determina se é obrigatório dividir o intervalo de comprimento c na posição c // 2

        Retorna
        -------
        ks             : list -- Lista de índices dos pontos de corte
        """

        # Calcula a folga do intervalo para a divisão esperada
        folga = c - f * t
        if folga <= 0:
            raise ValueError("Impossível dividir intervalo de comprimento {}"
                             " em {} fatias com {} de comprimento mínimo".format(c, f, t))

        # Calcula os índices como o resultado da soma cumulativa do vetor que contém o comprimento de cada
        # subintervalo tomado como o comprimento mínimo somado a uma distribuição aleatória da folga
        ks = np.cumsum([0] + list(np.array(f * [t]) + np.array(self.distribuir(folga, f))))

        # Corrige a divisão quando se deseja que haja um corte em c // 2
        if dividir_ao_meio:
            for i, k in enumerate(ks):
                if k >= c / 2:
                    if k == c / 2:
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

    @staticmethod
    def distribuir(folga, fatias):
        """
        Distribui a folga aleatoriamente dentre as fatias

        Argumentos
        ----------
        folga       : int  -- Tamanho da folga
        fatias      : int  -- Número de fatias

        Retorna
        -------
        distribuição: list -- Lista cujas posições contém a porção de folga que o intervalo correspondente recebeu
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

    @staticmethod
    def caminhar_até_a_borda(i_partida=0, j_partida=13):
        """
        Caminha aleatoriamente pelo grafo desde o ponto de aplicação da força até uma borda.

        Argumentos
        ----------
        i_partida: int       -- Índice da linha do grafo onde a caminhada começa
        j_partida: int       -- Índice da coluna do grafo onde a caminhada começa

        Retorna
        -------
        I        : List[int] -- Lista ordenada dos índices de linha de todos os nós do grafo por onde se passou
        J        : List[int] -- Lista ordenada dos índices de coluna de todos os nós do grafo por onde se passou
        """

        I = []
        J = []

        i = i_partida
        j = j_partida
        alcançou_a_borda = False
        último_movimento = "direita"

        # Começa a caminhada
        while not alcançou_a_borda:

            # Atualiza a trajetória
            I.append(i)
            J.append(j)

            # Inicializa a lista de possíveis próximos passos
            direções = ["cima", "baixo", "esquerda"]

            # Exclui as opções incoerentes com o último movimento
            if último_movimento == "cima":
                direções.remove("baixo")
            elif último_movimento == "baixo":
                direções.remove("cima")

            # Cuida para nque não se excedam as bordas
            if i == 0:
                direções.remove("cima")
            elif i == 6:
                direções.remove("baixo")

            # Determinar as probabilidades de se mover em
            # cada direção dada a quantidade de opções
            if len(direções) == 1:
                p = (1,)
            elif len(direções) == 2:
                p = (0.64, 0.36)
            else:
                p = (0.32, 0.32, 0.36)

            # Escolhe, finalmente, a direção de movimento
            mover_para = np.random.choice(direções, p=p)

            # Checa se a borda foi alcançada
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

        return I, J

    @staticmethod
    def formar_a_partir_do(grafo, kis, kjs):
        """
        Recupera a informação do grafo de partição do espaço de projeto para formar o gene.

        Argumentos
        ----------
        grafo: np.array(( 7, 14)) -- Grafo binário da partição do espaço de projeto
        kis  : List[int]          -- Índices de cortes horizontais do gene
        kjs  : List[int]          -- Índices de cortes verticais do gene

        Retorna
        -------
        gene : np.array((38, 76)) -- Gene de Projeto que satisfaz as restrições do problema
        """

        gene = np.zeros((38, 76), dtype=bool)

        for i in range(7):
            for j in range(14):
                gene[kis[i]:kis[i + 1], kjs[j]:kjs[j + 1]] = grafo[i][j]

        return gene

    # Ensina como cruzar os genes dos Indivíduos (Projetos) selecionados para reprodução
    def crossover(self, p1, p2, índice):
        """
        Gera um indivíduo filho a partir do cruzamento de dois indivíduos pais.

        Esta é uma implementação do crossover de 3 blocos. Faz-se dois cortes verticais e dois cortes horizontais
        em posições aleatórias dos genes dos indivíduos pais e se deriva o gene do indivíduo filho a partir de uma
        cópia do gene do pai 1 com 3 fatias quaisquer trocadas por suas correspondentes no gene do pai 2.

        Argumentos
        ----------
        p1    : Projeto -- Primeiro indivíduo pai
        p2    : Projeto -- Segundo indivíduo pai
        índice: int     -- Índice do filho na geração

        Retorna
        -------
        filho : Projeto -- Indivíduo filho
        """
        gene_novo = copy(p1.gene)

        # Seleciona 2 pontos de corte aleatórios em cada direção e garante que
        # eles sejam distintos e ordenados
        i1, i2, j1, j2 = np.random.randint(1, 37), np.random.randint(1, 37), \
                         np.random.randint(1, 75), np.random.randint(1, 75)
        while i1 == i2:
            i2 = np.random.randint(1, 37)
        while j1 == j2:
            j2 = np.random.randint(1, 75)

        i_b, i_c = max([i1, i2]), min([i1, i2])
        j_d, j_e = max([j1, j2]), min([j1, j2])

        # Delimita os intervalos das fatias em cada direção
        corte_horizontal = [(0, i_c), (i_c, i_b), (i_b, 76)]
        corte_vertical = [(0, j_e), (j_e, j_d), (j_d, 38)]

        # Seleciona 3 blocos aleatórios a partir das fatias
        blocos = []
        while len(blocos) < 3:
            bloco = (choice(corte_horizontal), choice(corte_vertical))
            if bloco not in blocos:
                blocos.append(bloco)

        # Escreve os 3 blocos retirados do gene do segundo projeto no gene novo
        for bloco in blocos:
            ibc, ibb = bloco[0]
            jbe, jbd = bloco[1]

            gene_novo[ibc:ibb, jbe:jbd] = p2.gene[ibc:ibb, jbe:jbd]

        return Projeto(gene_novo, "G{}_{}".format(self.geração, índice))

    def mutação(self, g):
        Médias = sum([ind.gene for ind in self.indivíduos]) / self.n
        Médias_2 = Médias ** 2

        for ind in g:
            probabilidade_de_mutar = self.pm + 99 * self.pm * Médias_2 + 99 * self.pm * ind.gene * (1 - 2 * Médias)
            mutações = probabilidade_de_mutar > np.random.random((38, 76))
            try:
                ind.gene[mutações] = virar_bit(ind.gene[mutações])
            except ValueError:
                pass

    def testar_adaptação(self, ind):
        ind.adaptação_testada = True

        l = 1 / 38

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
            indivíduos_selecionados = sorted(self.indivíduos, reverse=True)[:self.n // 2]

            genes = np.array([ind.gene.flatten() for ind in indivíduos_selecionados])

            rede_de_conexões = sch.linkage(genes, metric="correlation")
            espécies_dos_genes = sch.fcluster(rede_de_conexões, t=0.3, criterion="distance")
            espécies_iniciais = único(espécies_dos_genes)

            self.perfis_das_espécies = []
            for espécie in espécies_iniciais:
                genes_da_espécie = genes[espécies_dos_genes == espécie, :]
                perfil_da_espécie = np.mean(genes_da_espécie, axis=0)
                self.perfis_das_espécies[espécie] = perfil_da_espécie

            for i in range(len(indivíduos_selecionados)):
                indivíduos_selecionados[i].definir_espécie(espécies_dos_genes[i])

        indivíduos_selecionados = sorted(self.indivíduos, reverse=True)[:self.n // 2]

        return indivíduos_selecionados


virar_bit = np.vectorize(lambda b: 1 - b)

def único(vetor):
    """Retorna um vetor com os valores únicos do vetor original de forma ordenada"""
    _, índice = np.unique(vetor, return_index=True)
    return vetor[np.sort(índice)]


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
    gene           : np.array((38, 76)) -- Matriz binária que carrega o código genético
    l              : int                -- Comprimento do lado do elemento de membrana quadrado

    Retorna
    -------
    gene_útil      : np.array((38, 76)) -- Matriz binária que carrega a porção do gene que forma o fenótipo
    borda_alcançada: bool               -- O fenótipo se estende desde o ponto de aplicação da carga até a borda?
    elementos      : list               -- Lista de Elementos que compõem a malha
    nós            : list               -- Lista de Nós que compõem a malha
    me             : np.array(( 8, ne)) -- Matriz de correspondência entre os índices locais e globais de cada grau
                                           de liberdade (gsdl = graus de liberdade)
    """
    gene_útil = np.zeros((38, 76))

    # Define a posição inicial do algoritmo de busca
    i = 19
    j = 75

    elementos = []
    nós = []
    me = np.empty((8, 0), dtype="int16")

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
