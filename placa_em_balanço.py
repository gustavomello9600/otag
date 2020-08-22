"""
Este módulo utiliza os módulos de suporte para implementar classes que lidam com o problema
de determinar o melhor projeto para uma placa em balanço sujeita a uma carga puntiforme.

Seu objetivo é criar uma interface de uso do algoritmo genético implementado em suporte.algoritmo_genético para
otimizar topologicamente a placa em balanço discretizada pelo método dos elementos finitos e, por isso, entendida
como uma malha de membranas quadradas, objetos implementados em suporte.membrana_quadrada.

CONSTANTES
----------
CONSTANTE_DE_PENALIZAÇÃO_DA_ÁREA_DESCONECTADA: float -- Valor usado no cálculo de adaptação para penalizar a presença
                                                        de material genético não aproveitado

CLASSES
-------
Projeto
    Classe que carrega as propriedades de cada projeto. Herda seus principais atributos e métodos da classe
    Indivíduo do módulo algoritmo_genético.py
PopulaçãoDeProjetos
    Classe de objetos que agregam os indivíduos de uma população de projetos e os manipulam de acordo
    com operadores genéticos específicos ao problema. Herda seus atributos da classe População definida em
    suporte.algoritmo_genético.py e tem a maior parte dos seus métodos sobrescritos aqui.
"""

import math
from copy import copy
from random import choice, shuffle

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.sparse import csr_matrix

from suporte.elementos.membrana_quadrada import K_base, MembranaQuadrada
from suporte.algoritmo_genético import Indivíduo, População
from suporte.elementos_finitos import Nó, Malha, Problema


DESLOCAMENTO_LIMITE_DO_MATERIAL = 0.005
MÓDULO_DE_YOUNG_DO_MATERIAL     = 210e9
COEFICIENTE_DE_POYSSON          = 0.3
MAGNITUDE_DA_CARGA_APLICADA     = 100e6
ESPESSURA_DO_ELEMENTO           = 0.01
ORDEM_DE_REFINAMENTO_DA_MALHA   = 38

LADO_DO_ELEMENTO = 1/ORDEM_DE_REFINAMENTO_DA_MALHA

CONSTANTE_DE_PENALIZAÇÃO_DA_ÁREA_DESCONECTADA        = 0.4
CONSTANTE_DE_PENALIZAÇÃO_SOB_DESLOCAMENTO_EXCEDENTE  = 10


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
    espécie: int -- Número da espécie a qual o indivíduo pertence.
    """

    def __init__(self, *args, **kwargs):
        super(Projeto, self).__init__(*args, **kwargs)
        self._espécie = None

    def gerar_id_do_gene(self):
        return self.gene.data.tobytes()

    # Funcionalidade em Implementação
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
    suporte.algoritmo_genético.py e tem a maior parte dos seus métodos sobrescritos aqui.

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
        Executa todos os passos necessários para avançar uma geração como implementado pela classe População
        no módulo suporte.algoritmo_genético. Adicionalmente, atualiza o valor de alfa.
    geração_0(t: int) -> projetos: List[Projeto]
        Gera aleatoriamente 100 projetos de espessura interna mínima igual a t que estão conectados à borda
        e ao ponto de aplicação da carga.
    crossover(p1: Projeto, p2: Projeto, índice: int) -> Projeto
        Gera um indivíduo filho a partir do cruzamento de dois indivíduos pais.
    mutação(nova_geração: List[Projeto]) -> None
        Vira alguns bits dos genes dos indivíduos da próxima geração da população de acordo com uma
        probabilidade de mutar definida na construção da instância.
    testar_adaptação(ind: Projeto) -> None
        Testa a adaptação do indivíduo utilizando a modelagem e as condições de contorno do problema.
    seleção_natural() -> indivíduos_selecionados: List[Projeto]
        Seleciona os indivíduos com melhores genes.

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
    determinar_gene_útil(gene: np.array((38, 76)), l: int) -> gene_útil: np.array((38, 76)), borda_alcançada: bool,
                                                              elementos: List[Elemento], nós: List[Nó],
                                                              me: np.array((8, graus_de_liberdade))
        Executa um algoritmo de busca responsável por determinar, para um certo gene cuja expressão fenotípica é dada
        por uma malha de elementos quadrados de lado l, a maior porção contínua de matéria satisfazendo as restrições do
        problema, isto é, estar conectada simultaneamente ao ponto de aplicação da força e à borda.
    adicionar_à_malha_o_elemento_em(i: int, j: int, contexto: tuple) -> None
        Cria o elemento em i, j atualizando a matriz de correspondência entre índices globais e locais
    remover_de(possíveis_ramificações: list, i: int, j: int) -> None
        Caso seja uma candidata, remove a posição i, j da lista de possíveis ramificações da árvore de busca
    """

    def __init__(self, indivíduos=None, probabilidade_de_mutar=0.01 / 100):
        super(PopulaçãoDeProjetos, self).__init__(indivíduos=indivíduos,
                                                  probabilidade_de_mutar=probabilidade_de_mutar)
        self.placa_em_balanço = PlacaEmBalanço({"P": MAGNITUDE_DA_CARGA_APLICADA,
                                                "n": ORDEM_DE_REFINAMENTO_DA_MALHA}, método_padrão="OptV2_denso")

        # Em implementação
        self.perfis_das_espécies = dict()

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
        t: int -- Espessura interna mínima

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
            projetos.append(Projeto(gene, f"G0_{k + 1}"))

        return projetos

    @staticmethod
    def fatiar_intervalo(c=38, t=4, f=7, dividir_ao_meio=False):
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
            raise ValueError(f"Impossível dividir intervalo de comprimento "
                             f"{c} em {f} fatias com {t} de comprimento mínimo")

        # Abreviação do nome da classe usada para simplificar a chamada do método estático "distribuir"
        pdp = PopulaçãoDeProjetos

        # Calcula os índices como o resultado da soma cumulativa do vetor que contém o comprimento de cada
        # subintervalo tomado como o comprimento mínimo somado a uma distribuição aleatória da folga
        ks = np.cumsum([0] + list(np.array(f * [t]) + np.array(pdp.distribuir(folga, f))))

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

    def próxima_geração(self):
        self.placa_em_balanço.alfa = self.placa_em_balanço.alfa_0 * (1.01 ** (self.n_da_geração))
        super(PopulaçãoDeProjetos, self).próxima_geração()

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

        return Projeto(gene_novo, nome=f"G{self.n_da_geração}_{índice}")

    def mutação(self, nova_geração):
        """
        Vira alguns bits dos genes dos indivíduos da próxima geração da população corrente de acordo com uma
        probabilidade de mutar definida na construção da instância.

        Para cada bit de cada gene de cada indivíduo, calcula uma probabilidade de virar dependendo do seu valor
        correspondente na média dos genes da geração anterior. Um bit do indivíduo que já convergiu na população trará
        uma probabilidade mínima de virar caso esteja em concordância e uma probabilidade máxima caso esteja em discor-
        dância.

        Argumentos
        ----------
        nova_geração: List[Projeto] -- Lista de Projetos recém-formados pelo operador de crosover

        Retorna
        -------
        None
        """

        # Obtém a média, e a média ao quadrado, de cada bit na população
        Médias = sum([ind.gene for ind in self.indivíduos]) / self.n_de_indivíduos
        Médias_2 = Médias ** 2

        for ind in nova_geração:
            # Obtém a propabilidade de mutar de cada bit
            probabilidade_de_mutar = (self.probabilidade_de_mutar
                                      + 99 * self.probabilidade_de_mutar * Médias_2
                                      + 99 * self.probabilidade_de_mutar * ind.gene * (1 - 2 * Médias))

            # Sorteia os casos em que há mutação
            mutações = probabilidade_de_mutar > np.random.random((38, 76))

            # Vira os bits que resultaram em mutações
            ind.gene[mutações] = ~ind.gene[mutações]

    def testar_adaptação(self, ind):
        """
        Testa a adaptação do indivíduo utilizando a modelagem e as condições de contorno do problema.

        Argumentos
        ----------
        ind: Projeto -- Projeto que terá a adaptação determinada

        Retorna
        -------
        None
        """

        self.placa_em_balanço.testar_adaptação(ind)


class PlacaEmBalanço(Problema):

    fenótipos_testados = dict()

    Dlim   = DESLOCAMENTO_LIMITE_DO_MATERIAL
    alfa_0 = CONSTANTE_DE_PENALIZAÇÃO_SOB_DESLOCAMENTO_EXCEDENTE

    Monitorador = Problema.Monitorador

    def __init__(self, parâmetros_do_problema, método_padrão=None):
        super().__init__(parâmetros_do_problema, método_padrão)

        self.alfa = self.alfa_0
        self.Ke = None
        self._montador_do = {"expansão": self.montador_expansão,
                             "compacto": self.montador_compacto,
                             "OptV1": self.montador_OptV1,
                             "OptV2": self.montador_OptV2,
                             "OptV2_denso": self.montador_OptV2_denso}

        print("Classe Nó incrementada com novo método")

    # Métodos que recebem um indivíduo e retornam sua adaptação
    def testar_adaptação(self, ind):
        """
        Invoca um método que constrói o fenótipo do indivíduo, isto é, sua malha, a partir da porção útil do gene. Caso
        a malha não esteja conectada à borda, atribui adaptação 0 ao indivíduo e sinaliza na saída do sistema. Caso es-
        teja, verifica se um fenótipo idêntico já teve sua adaptação calculada. Caso não tenha, aplica o cálculo da a-
        daptação.
        """

        # Carrega o lado, em metros, do elemento de membrana quadrada
        l = LADO_DO_ELEMENTO

        # Chama o algoritmo de identificação da porção útil do gene e construção do fenótipo.
        fenótipo, borda_alcançada, elementos_conectados, nós, me = self.determinar_fenótipo(ind.gene, l)

        if not borda_alcançada:
            print(f"> Indivíduo {ind.nome} desconectado da borda")
            ind.adaptação = 0

        else:
            # Checa se este fenótipo já teve sua adaptação calculada antes
            if fenótipo.data.tobytes() in self.fenótipos_testados:

                # Recupera a adaptação do cache
                ind.adaptação = self.fenótipos_testados[fenótipo.data.tobytes()]
                print(f"> Adaptação de {ind.nome} já era conhecida pelo seu fenótipo")

            else:
                # Determina que os tempos de execução de cada etapa da análise por elementos
                # finitos sejam mensurados cada vez que o nome do Projeto terminar em "1"
                monitorar = ind.nome.endswith("1")

                ind.f, ind.u, ind.malha = \
                    self.resolver_para(

                        monitorar=monitorar,
                        malha=Malha(elementos_conectados, nós, me),
                        parâmetros_dos_elementos={"l": LADO_DO_ELEMENTO,
                                                  "t": ESPESSURA_DO_ELEMENTO,
                                                  "v": COEFICIENTE_DE_POYSSON,
                                                  "E": MÓDULO_DE_YOUNG_DO_MATERIAL}

                    )

                # Determina as áreas conectadas e desconectadas
                Acon = fenótipo.sum() * (l ** 2)
                Ades = ind.gene.sum() * (l ** 2) - Acon

                # Calcula o deslocamento máximo como a raiz quadrada do maior
                # valor de u_x² + u_y² dentre todos os nós da malha
                n = len(nós)
                Dmax = np.sqrt(np.sum(ind.u.reshape((n, 2)) ** 2, axis=1).max())

                penalização = Dmax - self.Dlim if Dmax > self.Dlim else 0

                if penalização:
                    print(f"> Indivíduo {ind.nome} penalizado: Dmax - Dlim = {penalização:.3e} metros")

                e = CONSTANTE_DE_PENALIZAÇÃO_DA_ÁREA_DESCONECTADA
                ind.adaptação = 1 / (Acon + e * Ades + self.alfa * penalização)

                print(f"> {ind.nome} conectado à borda. Adaptação: {ind.adaptação}")

        ind.adaptação_testada = True

    @staticmethod
    def determinar_fenótipo(gene, l):
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
        gene_útil = np.zeros((38, 76), dtype=bool)

        # Define a posição inicial do algoritmo de busca
        i = 19
        j = 75

        elementos = []
        nós = []
        me = []

        # Inicializa listas auxiliares que ajudam a manter curso dos índices dos nós
        etiquetas_de_nós_já_construídos = set()
        índice_na_malha = dict()

        # Inicializa parâmetros do algoritmo de busca
        possíveis_ramificações = []
        último_movimento = "esquerda"
        borda_alcançada = False
        buscando = True
        descida = True
        subida = False

        # Junta as variáveis importantes para métodos auxiliares
        contexto = l, gene_útil, elementos, nós, me, etiquetas_de_nós_já_construídos, índice_na_malha

        # Simplifica a chamada de métodos auxiliares
        peb = PlacaEmBalanço

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

                peb.adicionar_à_malha_o_elemento_em(i, j, contexto=contexto)
                peb.remover_de(possíveis_ramificações, i, j)

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

                peb.adicionar_à_malha_o_elemento_em(i, j, contexto=contexto)
                peb.remover_de(possíveis_ramificações, i, j)

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

        # Inicializa os dados em matriz
        me = np.array(me, dtype="int16").T

        return gene_útil, borda_alcançada, elementos, nós, me

    @staticmethod
    def adicionar_à_malha_o_elemento_em(i, j, contexto=tuple()):
        # Recebe o contexto
        l, gene_útil, elementos, nós, me, etiquetas_de_nós_já_construídos, índice_na_malha = contexto

        # Marca a posição como pertencente ao gene útil
        gene_útil[i][j] = True

        # Inicializa os nós dos cantos do elemento
        y = 1 - i * l
        ul, ur, dr, dl = Nó(j * l, y, etiqueta=(i, j)), \
                         Nó((j + 1) * l, y, etiqueta=(i, j + 1)), \
                         Nó((j + 1) * l, y - l, etiqueta=(i + 1, j + 1)), \
                         Nó(j * l, y - l, etiqueta=(i + 1, j))

        índices_globais_dos_cantos = []

        # Para cada nó em cada canto
        for nó in (ul, ur, dr, dl):

            # Se o índice do nó ainda não foi visto
            if nó.etiqueta not in etiquetas_de_nós_já_construídos:

                # Adiciona o canto à lista de nós que comporão a malha
                nós.append(nó)

                # Define o índice do canto na malha como o índice corrente
                índice_na_malha[nó.etiqueta] = len(etiquetas_de_nós_já_construídos)
                índices_globais_dos_cantos.append(len(etiquetas_de_nós_já_construídos))

                # Define que o nó já foi visto
                etiquetas_de_nós_já_construídos.add(nó.etiqueta)

            else:
                índices_globais_dos_cantos.append(índice_na_malha[nó.etiqueta])

        iul, iur, idr, idl = índices_globais_dos_cantos

        # Atualiza a matriz de correspondência entre os índices
        # locais e globais dos nós para o elemento atual
        me.append((2 * iul,
                   2 * iul + 1,
                   2 * iur,
                   2 * iur + 1,
                   2 * idr,
                   2 * idr + 1,
                   2 * idl,
                   2 * idl + 1))

        # Cria um elemento com os cantos e o adiciona
        # à lista daqueles que comporão a malha
        elementos.append(MembranaQuadrada([ul, ur, dr, dl]))

    @staticmethod
    def remover_de(possíveis_ramificações, i, j):
        try:
            possíveis_ramificações.remove((i, j, "esquerda"))
        except ValueError:
            try:
                possíveis_ramificações.remove((i, j, "direita"))
            except ValueError:
                pass

    # Métodos auxiliares da resolução via análise de elementos finitos
    @Monitorador(mensagem="Total de graus de liberdade determinados")
    def determinar_graus_de_liberdade(self, malha):
        return 2*len(malha.nós)

    @Monitorador(mensagem="Matrizes de rigidez local determinadas")
    def calcular_matrizes_de_rigidez_local(self, **parâmetros_do_elemento_base):
        if self.Ke is None:
            self.Ke = K_base.calcular(parâmetros_do_elemento_base)

            l = parâmetros_do_elemento_base['l']
            t = parâmetros_do_elemento_base['t']
            v = parâmetros_do_elemento_base['v']
            E = parâmetros_do_elemento_base['E']

            assert math.isclose(self.Ke[0, 0], (2*t*E*(v - 3))/(3 * (l**2) * (v**2 - 1)))

        return self.Ke

    @Monitorador(mensagem="Matriz de rigidez global montada")
    def montar_matriz_de_rigidez_geral(self, malha, Ke, graus_de_liberdade, método):
        return self._montador_do[método](malha, Ke, graus_de_liberdade)

    @staticmethod
    def montador_expansão(malha, Ke, graus_de_liberdade):
        Kes_expandidos = dict()
        for elemento in malha.elementos:
            Ke_expandido = np.zeros((graus_de_liberdade, graus_de_liberdade))

            índices = np.array([
                [2 * malha.índice_de(n), 2 * malha.índice_de(n) + 1] for n in elemento.nós
            ]).flatten()

            for ie in range(len(índices)):
                for je in range(len(índices)):
                    i = índices[ie]
                    j = índices[je]

                    Ke_expandido[i][j] = Ke[ie][je]

            Kes_expandidos[elemento] = Ke_expandido

        return sum(Kes_expandidos.values())

    @staticmethod
    def montador_compacto(malha, Ke, graus_de_liberdade):
        K = np.zeros((graus_de_liberdade, graus_de_liberdade))

        índices = dict()
        for elemento in malha.elementos:
            índices[elemento] = np.array([[2 * malha.índice_de(n), 2 * malha.índice_de(n) + 1]
                                          for n in elemento.nós]).flatten()

        for i in range(8):
            for j in range(8):
                for e in malha.elementos:
                    p = índices[e][i]
                    q = índices[e][j]

                    K[p][q] += Ke[i][j]

        return K

    @staticmethod
    def montador_OptV1(malha, Ke, graus_de_liberdade):
        D = np.zeros(64 * malha.ne)
        I = np.zeros(64 * malha.ne, dtype="int32")
        J = np.zeros(64 * malha.ne, dtype="int32")
        d = 0

        índices_de_Ke_por_elemento = ((e, i, j) for e in range(malha.ne)
                                                for i in range(8)
                                                for j in range(8))

        for d, (e, i, j) in enumerate(índices_de_Ke_por_elemento):
            I[d] = malha.me[i][e]
            J[d] = malha.me[j][e]
            D[d] = Ke[i][j]

        return csr_matrix((D, (I, J)), shape=(graus_de_liberdade, graus_de_liberdade)).toarray()

    @staticmethod
    def montador_OptV2(malha, Ke, graus_de_liberdade):
        D = np.zeros((64, malha.ne), dtype=float)
        I = np.zeros((64, malha.ne), dtype="int32")
        J = np.zeros((64, malha.ne), dtype="int32")

        índices_de_Ke = ((i, j) for i in range(8) for j in range(8))

        for d, (i, j) in enumerate(índices_de_Ke):
            D[d, :] = np.repeat(Ke[i, j], malha.ne)
            I[d, :] = malha.me[i, :]
            J[d, :] = malha.me[j, :]

        return csr_matrix((D.flat, (I.flat, J.flat)), shape=(graus_de_liberdade, graus_de_liberdade)).toarray()

    @staticmethod
    def montador_OptV2_denso(malha, Ke, graus_de_liberdade):
        K = np.empty((graus_de_liberdade, graus_de_liberdade), dtype=float)

        índices_de_Ke = ((i, j) for i in range(8) for j in range(8))

        for i, j in índices_de_Ke:
            K[malha.me[i, :], malha.me[j, :]] = Ke[i, j]

        return K

    @Monitorador(mensagem="Condições de contorno incorporadas")
    def incorporar_condições_de_contorno(self, malha, graus_de_liberdade, P=0e0, n=1):

        f = np.zeros(graus_de_liberdade)
        u = np.full(graus_de_liberdade, np.nan)

        # Condições de Contorno em u
        for i in range(n + 1):
            try:
                i1 = 2 * malha.nós.index(Nó(0, 1 - i / n))
            except ValueError:
                continue
            i2 = i1 + 1
            u[i1:(i2 + 1)] = 0
            f[i1:(i2 + 1)] = np.nan

        # Condições de Contorno em f
        gdl_P = grau_de_liberdade_associado_a_P = malha.nós.index(Nó(2, 0.5)) * 2 + 1
        f[gdl_P] = -P

        ifc = índices_onde_f_é_conhecido = np.where(~np.isnan(f))[0]
        iuc = índices_onde_u_é_conhecido = np.where(~np.isnan(u))[0]

        return f, u, ifc, iuc


# TODO Implementação primitiva de um classificador de espécies
def implementar_dentro():
    def do_método_seleção_natural(self):
        if self.perfis_das_espécies is None:
            indivíduos_selecionados = sorted(self.indivíduos, reverse=True)[:self.n_de_indivíduos // 2]

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


def único(vetor):
    """Retorna um vetor com os valores únicos do vetor original de forma ordenada"""
    _, índice = np.unique(vetor, return_index=True)
    return vetor[np.sort(índice)]
