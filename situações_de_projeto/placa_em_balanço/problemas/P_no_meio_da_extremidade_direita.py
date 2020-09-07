import math
from collections import OrderedDict
from random import choice, shuffle

import numpy as np

from suporte.elementos_finitos import Malha, Nó
from suporte.elementos_finitos.definição_de_problema import Problema
from suporte.elementos_finitos.membrana_quadrada import MembranaQuadrada, K_base


class PlacaEmBalanço(Problema):
    """
    Implementação do problema da Placa em Balanço 2x1

    Métodos
    -------
    determinar_gene_útil(gene: np.array((38, 76)), l: int) -> gene_útil: np.array((38, 76)), borda_alcançada: bool,
                                                              elementos: List[Elemento], nós: List[Nó],
                                                              me: np.array((8, ne))
        Executa um algoritmo de busca responsável por determinar, para um certo gene cuja expressão fenotípica é dada
        por uma malha de elementos quadrados de lado l, a maior porção contínua de matéria satisfazendo as restrições do
        problema, isto é, estar conectada simultaneamente ao ponto de aplicação da força e à borda.
    adicionar_à_malha_o_elemento_em(i: int, j: int, contexto: tuple) -> None
        Cria o elemento em i, j atualizando a matriz de correspondência entre índices globais e locais.
    remover_de(possíveis_ramificações: list, i: int, j: int) -> None
        Caso seja uma candidata, remove a posição i, j da lista de possíveis ramificações da árvore de busca.


    ESTÁTICOS (AUXILIARES)
    ---------
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

    Monitorador = Problema.Monitorador

    def __init__(self, parâmetros_do_problema, método_padrão=None):
        # Inicia um cache de fenótipos
        self.fenótipos_testados = Cache(maxsize=300)

        super().__init__(parâmetros_do_problema, método_padrão)
        self._digerir(parâmetros_do_problema)
        self._iniciar_resolvedor()

    def _digerir(self, parâmetros_do_problema):
        self.Dlim = parâmetros_do_problema["DESLOCAMENTO_LIMITE_DO_MATERIAL"]
        self.lado_dos_elementos = 1 / parâmetros_do_problema["ORDEM_DE_REFINAMENTO_DA_MALHA"]
        self._método_padrão = parâmetros_do_problema["MÉTODO_PADRÃO_DE_MONTAGEM_DA_MATRIZ_DE_RIGIDEZ_GERAL"]
        self.alfa = self.alfa_0 = parâmetros_do_problema["CONSTANTE_DE_PENALIZAÇÃO_SOB_DESLOCAMENTO_EXCEDENTE"]
        self.e = parâmetros_do_problema["CONSTANTE_DE_PENALIZAÇÃO_DA_ÁREA_DESCONECTADA"]

    def _iniciar_resolvedor(self):
        self.Ke = None
        self._montador_do = {"expansão": self.montador_expansão,
                             "compacto": self.montador_compacto,
                             "OptV1": self.montador_OptV1,
                             "OptV2": self.montador_OptV2}

    def geração_0(self, n_de_indivíduos=125, t=4):
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

        genes = []
        for k in range(n_de_indivíduos):
            # Inicia um grafo que representa o preenchimento de cada fatia do espaço de projeto
            grafo = np.random.choice((True, False), (7, 14))

            # Determina os pontos de corte e retorna seus índices i e j
            kis = self._fatiar_intervalo(c=38, t=t, f=7, dividir_ao_meio=True)
            kjs = self._fatiar_intervalo(c=76, t=t, f=14, dividir_ao_meio=False)

            # Caminha aleatoriamente pelo grafo desde o ponto de aplicação da força até uma borda,
            # preenchendo as fatias ao longo da trajetória
            trajetória = self._caminhar_até_a_borda(i_partida=int(np.where(kis == 19)[0]), j_partida=13)
            grafo[trajetória[0], trajetória[1]] = 1

            gene = self._formar_gene_a_partir_do(grafo, kis, kjs)

            genes.append(gene)

        return genes

    @staticmethod
    def _fatiar_intervalo(c=38, t=4, f=7, dividir_ao_meio=False):
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
        peb = PlacaEmBalanço

        # Calcula os índices como o resultado da soma cumulativa do vetor que contém o comprimento de cada
        # subintervalo tomado como o comprimento mínimo somado a uma distribuição aleatória da folga
        ks = np.cumsum([0] + list(np.array(f * [t]) + np.array(peb._distribuir(folga, f))))

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
    def _distribuir(folga, fatias):
        """
        Distribui a folga aleatoriamente dentre as fatias

        Argumentos
        ----------
        folga : int -- Tamanho da folga
        fatias: int -- Número de fatias

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
    def _caminhar_até_a_borda(i_partida=0, j_partida=13):
        """
        Caminha aleatoriamente pelo grafo desde o ponto de aplicação da força até uma borda.

        Argumentos
        ----------
        i_partida: int -- Índice da linha do grafo onde a caminhada começa
        j_partida: int -- Índice da coluna do grafo onde a caminhada começa

        Retorna
        -------
        I: List[int] -- Lista ordenada dos índices de linha de todos os nós do grafo por onde se passou
        J: List[int] -- Lista ordenada dos índices de coluna de todos os nós do grafo por onde se passou
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

            # Cuida para que não se excedam as bordas
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
    def _formar_gene_a_partir_do(grafo, kis, kjs):
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

    def testar_adaptação(self, ind):
        """
        Invoca um método que constrói o fenótipo do indivíduo, isto é, sua malha, a partir da porção útil do gene. Caso
        a malha não esteja conectada à borda, atribui adaptação 0 ao indivíduo e sinaliza na saída do sistema. Caso es-
        teja, verifica se um fenótipo idêntico já teve sua adaptação calculada. Caso não tenha, aplica o cálculo da a-
        daptação.
        """

        # Carrega o lado, em metros, do elemento de membrana quadrada
        l = self.lado_dos_elementos

        # Chama o algoritmo de identificação da porção útil do gene e construção do fenótipo.
        fenótipo, borda_alcançada, elementos_conectados, nós, me = self._determinar_fenótipo(ind.gene, l)

        if not self._atende_os_requisitos_mínimos(fenótipo, borda_alcançada):
            print(f"> Indivíduo {ind.nome} desconectado da borda")
            ind.adaptação = 0

        else:
            # Checa se este fenótipo já teve sua adaptação calculada antes
            if fenótipo.data.tobytes() in self.fenótipos_testados:

                # Recupera a adaptação do cache
                ind.adaptação, ind.f, ind.u, ind.malha = self.fenótipos_testados[fenótipo.data.tobytes()]
                print(f"> Adaptação de {ind.nome} já era conhecida pelo seu fenótipo")

            else:
                # Determina que os tempos de execução de cada etapa da análise por elementos_finitos
                # finitos sejam mensurados cada vez que o nome do Projeto terminar em "1"
                monitorar = ind.nome.endswith("1")

                ind.f, ind.u, ind.malha = \
                    self.resolver_para(

                        monitorar=monitorar,
                        malha=Malha(elementos_conectados, nós, me),
                        parâmetros_dos_elementos={"l": l,
                                                  "t": self.parâmetros_do_problema["ESPESSURA_DO_ELEMENTO"],
                                                  "v": self.parâmetros_do_problema["COEFICIENTE_DE_POYSSON"],
                                                  "E": self.parâmetros_do_problema["MÓDULO_DE_YOUNG_DO_MATERIAL"]}

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

                ind.adaptação = 1 / (Acon + self.e * Ades + self.alfa * penalização)

                print(f"> {ind.nome} conectado à borda. Adaptação: {ind.adaptação}")

                self.fenótipos_testados[fenótipo.data.tobytes()] = ind.adaptação, ind.f, ind.u, ind.malha

        ind.adaptação_testada = True

    @staticmethod
    def _determinar_fenótipo(gene, l):
        """
        Executa um algoritmo de busca responsável por determinar, para um certo gene cuja expressão fenotípica é dada
        por uma malha de elementos_finitos quadrados de lado l, a maior porção contínua de matéria satisfazendo as res-
        trições do problema, isto é, estar conectada simultaneamente ao ponto de aplicação da força e à borda.

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
                                               de liberdade
        """
        gene_útil = np.zeros((38, 76), dtype=bool)

        # Define a posição inicial do algoritmo de busca
        i = 19
        j = 75

        elementos = []
        nós = []
        me = []

        # Inicializa listas auxiliares que ajudam a manter curso dos índices dos nós e elementos
        etiquetas_de_elementos_já_construídos = set()
        etiquetas_de_nós_já_construídos = set()
        índice_na_malha = dict()

        # Inicializa parâmetros do algoritmo de busca
        possíveis_ramificações = set()
        último_movimento = "esquerda"
        borda_alcançada = False
        buscando = True
        descida = True
        subida = False

        # Junta as variáveis importantes para métodos auxiliares
        contexto = l, gene_útil, elementos, nós, me, etiquetas_de_nós_já_construídos, \
                   etiquetas_de_elementos_já_construídos, índice_na_malha

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
                        possíveis_ramificações.add((i, j + 1, "direita"))

                if j == 0:
                    borda_alcançada = True

                if j != 0 and último_movimento != "direita":
                    esquerda = gene[i][j - 1]
                    if esquerda and not gene_útil[i][j - 1]:
                        possíveis_ramificações.add((i, j - 1, "esquerda"))

                peb._adicionar_à_malha_o_elemento_em(i, j, contexto=contexto)
                peb._remover_de(possíveis_ramificações, i, j)

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
                        possíveis_ramificações.add((i, j + 1, "direita"))

                if j == 0:
                    borda_alcançada = True

                if j != 0 and último_movimento != "direita":
                    esquerda = gene[i][j - 1]
                    if esquerda and not gene_útil[i][j - 1]:
                        possíveis_ramificações.add((i, j - 1, "esquerda"))

                peb._adicionar_à_malha_o_elemento_em(i, j, contexto=contexto)
                peb._remover_de(possíveis_ramificações, i, j)

                # Decide se continua descendo ou se passa a subir
                if subida:
                    i = i - 1
                    último_movimento = "cima"

            if len(possíveis_ramificações) > 0:
                i, j, último_movimento = possíveis_ramificações.pop()
                descida = True
                subida = False

            else:
                buscando = False

        # Transforma a lista de tuplas em matriz
        me = np.array(me, dtype="int16").T

        return gene_útil, borda_alcançada, elementos, nós, me

    @staticmethod
    def _adicionar_à_malha_o_elemento_em(i, j, contexto):
        # Recebe o contexto
        l, gene_útil, elementos, nós, me, etiquetas_de_nós_já_construídos, \
        etiquetas_de_elementos_já_construídos, índice_na_malha               = contexto

        # Marca a posição como pertencente ao gene útil
        gene_útil[i][j] = True

        # Inicializa os nós dos cantos do elemento
        y = 1 - i * l
        ul, ur, dr, dl = Nó(      j*l,     y, etiqueta=(    i,     j)), \
                         Nó((j + 1)*l,     y, etiqueta=(    i, j + 1)), \
                         Nó((j + 1)*l, y - l, etiqueta=(i + 1, j + 1)), \
                         Nó(      j*l, y - l, etiqueta=(i + 1,     j))

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
        if ul.etiqueta not in etiquetas_de_elementos_já_construídos:
            elementos.append(MembranaQuadrada((ul, ur, dr, dl)))
            etiquetas_de_elementos_já_construídos.add(ul.etiqueta)

    @staticmethod
    def _remover_de(possíveis_ramificações, i, j):
        possíveis_ramificações.discard((i, j, "esquerda"))
        possíveis_ramificações.discard((i, j, "direita"))

    @staticmethod
    def _atende_os_requisitos_mínimos(fenótipo, borda_alcançada):
        return borda_alcançada

    # Métodos auxiliares da resolução via análise de elementos finitos
    @Monitorador(mensagem="Total de graus de liberdade determinados")
    def determinar_graus_de_liberdade(self, malha):
        return 2 * len(malha.nós)

    @Monitorador(mensagem="Matrizes de rigidez local determinadas")
    def calcular_matrizes_de_rigidez_local(self, **parâmetros_do_elemento_base):
        if self.Ke is None:
            self.Ke = K_base.calcular(parâmetros_do_elemento_base)

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
                [2 * malha.índice_de[n], 2 * malha.índice_de[n] + 1] for n in elemento.nós
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
            índices[elemento] = np.array([[2 * malha.índice_de[n], 2 * malha.índice_de[n] + 1]
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
        K = np.zeros((graus_de_liberdade, graus_de_liberdade), dtype=float)

        índices_de_Ke_por_elemento = ((e, i, j) for e in range(malha.ne)
                                      for i in range(8)
                                      for j in range(8))

        for d, (e, i, j) in enumerate(índices_de_Ke_por_elemento):
            K[malha.me[i][e], malha.me[j][e]] += Ke[i][j]

        return K

    @staticmethod
    def montador_OptV2(malha, Ke, graus_de_liberdade):
        K = np.zeros((graus_de_liberdade, graus_de_liberdade), dtype=float)

        índices_de_Ke = ((i, j) for i in range(8) for j in range(8))

        for i, j in índices_de_Ke:
            K[malha.me[i, :], malha.me[j, :]] += Ke[i, j]

        return K

    @Monitorador(mensagem="Condições de contorno incorporadas")
    def incorporar_condições_de_contorno(self, malha, graus_de_liberdade, parâmetros_do_problema):
        P = parâmetros_do_problema["MAGNITUDE_DA_CARGA_APLICADA"]
        n = parâmetros_do_problema["ORDEM_DE_REFINAMENTO_DA_MALHA"]

        f = np.zeros(graus_de_liberdade)
        u = np.full(graus_de_liberdade, np.nan)

        # Condições de Contorno em u
        for i in range(n + 1):
            nó_da_borda = Nó(0, 1 - i/n, etiqueta=(i, 0))
            if nó_da_borda in malha.índice_de:
                i1 = 2*malha.índice_de[nó_da_borda]
                i2 = i1 + 1
                u[[i1, i2]] = 0
                f[[i1, i2]] = np.nan

        # Condições de Contorno em f
        gdl_P = grau_de_liberdade_associado_a_P = malha.nós.index(Nó(2, 0.5)) * 2 + 1
        f[gdl_P] = -P

        ifc = índices_onde_f_é_conhecido = np.where(~np.isnan(f))[0]
        iuc = índices_onde_u_é_conhecido = np.where(~np.isnan(u))[0]

        return f, u, ifc, iuc


class Cache(OrderedDict):
    """Cache de valores recentemente usados"""

    def __init__(self, maxsize=128, **kwds):
        self.maxsize = maxsize
        super().__init__(**kwds)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]
