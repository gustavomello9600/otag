from itertools import product as produto_cartesiano
from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Optional, Callable
import random

import numpy as np

from suporte.elementos_finitos import Malha, Nó, Matriz, Vetor
from suporte.elementos_finitos.definição_de_problema import Problema, Máscara
from suporte.elementos_finitos.membrana_quadrada import MembranaQuadrada, K_base


Gene = Matriz
FunçãoMontadora = Callable[[Malha, Matriz, int], Matriz]


class PlacaEmBalanço(Problema):
    """Implementação do problema da Placa em Balanço 2x1"""

    Monitorador = Problema.Monitorador

    def __init__(self, parâmetros_do_problema, método_padrão=None):
        # Inicia um cache de fenótipos
        self.fenótipos_testados = Cache(maxsize=300)

        super().__init__(parâmetros_do_problema, método_padrão)
        self._digerir(parâmetros_do_problema)
        self._iniciar_resolvedor()

    def _digerir(self, parâmetros_do_problema):
        self.n: int = parâmetros_do_problema["ORDEM_DE_REFINAMENTO_DA_MALHA"]
        if self.n % 2 != 0:
            raise ValueError(f"A ordem de refinamento da malha deve ser par. {self.n} fornecido.")
        if self.n < 7:
            raise ValueError(f"A ordem de refinamento da malha deve ser maior que 7. {self.n} fornecido.")

        self._método_padrão    : str   = parâmetros_do_problema["MÉTODO_PADRÃO_DE_MONTAGEM_DA_MATRIZ_DE_RIGIDEZ_GERAL"]
        self.Dlim              : float = parâmetros_do_problema["DESLOCAMENTO_LIMITE_DO_MATERIAL"]
        self.alfa_0            : float = parâmetros_do_problema["CONSTANTE_DE_PENALIZAÇÃO_SOB_DESLOCAMENTO_EXCEDENTE"]
        self.e                 : float = parâmetros_do_problema["CONSTANTE_DE_PENALIZAÇÃO_DA_ÁREA_DESCONECTADA"]

        self.lado_dos_elementos = 1/self.n
        self.alfa = self.alfa_0

    def _iniciar_resolvedor(self):
        self.Ke: Optional[Matriz] = None
        self._montador_do: Dict[str, FunçãoMontadora] = {
            "expansão": self.montador_expansão,
            "compacto": self.montador_compacto,
            "OptV1": self.montador_OptV1,
            "OptV2": self.montador_OptV2
        }

    def geração_0(self, n_de_indivíduos: int = 125, espessura_interna_mínima: int = 4) -> List[Gene]:
        """
        Gera aleatoriamente 100 projetos de espessura interna mínima igual a t que estão conectados à borda
        e ao ponto de aplicação da carga.

        Fatia o espaço de projeto em 7 x 14 pedaços interpretados como nós de um grafo. Inicialmente, atribui
        um valor binário qualquer a cada nó e em seguida executa um algoritmo simples de caminhada aleatória desde
        o ponto de aplicação da carga até a borda. Os nós pertencentes à trajetória percorrida no grafo recebem o valor
        1. O grafo é então traduzido para um gene que inicia uma instância da classe Projeto. Este processo é repetido
        e o resultado de 100 iterações é retornado numa lista.
        """

        genes = []
        for k in range(n_de_indivíduos):
            # Inicia um grafo que representa o preenchimento de cada fatia do espaço de projeto
            grafo = np.random.choice((True, False), (7, 14))

            # Determina os pontos de corte e retorna seus índices i e j
            pontos_de_corte_horizontal = self._fatiar_intervalo(comprimento_total=self.n,
                                                                número_de_fatias=7,
                                                                dividir_ao_meio=True,
                                                                comprimento_mínimo_das_fatias=espessura_interna_mínima)
            pontos_de_corte_vertical = self._fatiar_intervalo(comprimento_total=2*self.n,
                                                              número_de_fatias=14,
                                                              dividir_ao_meio=False,
                                                              comprimento_mínimo_das_fatias=espessura_interna_mínima)

            # Caminha aleatoriamente pelo grafo desde o ponto de aplicação da força até uma borda,
            # preenchendo as fatias ao longo da trajetória
            trajetória = self._caminhar_até_a_borda(
                i_partida=int(np.where(pontos_de_corte_horizontal == int(self.n//2))[0]),
                j_partida=13
            )
            grafo[trajetória[0], trajetória[1]] = 1

            gene = self._formar_gene_a_partir_do(grafo,
                                                 pontos_de_corte_vertical=pontos_de_corte_vertical,
                                                 pontos_de_corte_horizontal=pontos_de_corte_horizontal)

            genes.append(gene)

        return genes

    @staticmethod
    def _fatiar_intervalo(comprimento_total: int = 38, número_de_fatias: int = 7,
                          comprimento_mínimo_das_fatias: int = 4, dividir_ao_meio: bool = False
                          ) -> Vetor:
        """Fatia aleatoriamente um intervalo de comprimento c em f fatias (ou subintervalos) de comprimento mínimo t
        e retorna um vetor com números inteiros correspondentes aos índices dos pontos de corte."""

        # Calcula a folga do intervalo para a divisão esperada
        folga = comprimento_total - número_de_fatias * comprimento_mínimo_das_fatias
        if folga <= 0:
            raise ValueError(f"Impossível dividir intervalo de comprimento {comprimento_total} em "
                             f"{número_de_fatias} fatias com {comprimento_mínimo_das_fatias} de comprimento mínimo")

        # Abreviação do nome da classe usada para simplificar a chamada de métodos estáticos
        peb = PlacaEmBalanço

        # Calcula os índices como o resultado da soma cumulativa do vetor que contém o comprimento de cada
        # subintervalo tomado como o comprimento mínimo somado a uma distribuição aleatória da folga
        índices_dos_pontos_de_corte = np.cumsum([0] + list(np.array(número_de_fatias * [comprimento_mínimo_das_fatias])
                                                           + np.array(peb._distribuir(folga, número_de_fatias))))

        # Corrige a divisão quando se deseja que haja um corte em c // 2
        if dividir_ao_meio:
            peb._adaptar_divisão(comprimento_total, índices_dos_pontos_de_corte, comprimento_mínimo_das_fatias)

        return índices_dos_pontos_de_corte

    @staticmethod
    def _distribuir(folga: int, número_de_fatias: int) -> List[int]:
        """Distribui a folga aleatoriamente dentre as fatias"""

        distribuição = []
        for _ in range(número_de_fatias):
            espaço_extra = random.choice(list(range(folga + 1)))
            folga -= espaço_extra
            distribuição.append(espaço_extra)

        if folga > 0:
            distribuição[-1] += folga

        random.shuffle(distribuição)
        return distribuição

    @staticmethod
    def _adaptar_divisão(comprimento_total: int, pontos_de_corte: Vetor, comprimento_mínimo_das_fatias: int) -> None:
        for i, ponto_de_corte in enumerate(pontos_de_corte):
            if ponto_de_corte >= comprimento_total / 2:
                if ponto_de_corte == comprimento_total / 2:
                    break
                pontos_de_corte[i - 1] = comprimento_total / 2
                j = i
                while pontos_de_corte[j] - pontos_de_corte[j - 1] < comprimento_mínimo_das_fatias:
                    pontos_de_corte[j] += comprimento_mínimo_das_fatias - (pontos_de_corte[j] - pontos_de_corte[j - 1])
                    if j == len(pontos_de_corte) - 1:
                        j = 1
                    else:
                        j += 1
                break

    @staticmethod
    def _caminhar_até_a_borda(i_partida: int = 0, j_partida: int = 13) -> Tuple[List[int], List[int]]:
        """Caminha aleatoriamente pelo grafo desde o ponto de aplicação da força até uma borda.̉"""

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

    def _formar_gene_a_partir_do(self,
                                 grafo: Matriz,
                                 pontos_de_corte_vertical: Vetor,
                                 pontos_de_corte_horizontal: Vetor
                                 ) -> Gene:
        """Recupera a informação do grafo de partição do espaço de projeto para formar o gene."""

        gene = np.zeros((self.n, 2*self.n), dtype=bool)

        for i, j in produto_cartesiano(range(7), range(14)):
            gene[pontos_de_corte_horizontal[i]:pontos_de_corte_horizontal[i + 1],
                 pontos_de_corte_vertical[j]:pontos_de_corte_vertical[j + 1]] = grafo[i][j]

        return gene

    def testar_adaptação(self, proj: 'Projeto') -> None:
        """
        Constrói e testa o fenótipo do projeto.

        Caso a malha não esteja conectada à borda, atribui adaptação 0 ao indivíduo e sinaliza na saída do sistema. Caso
        esteja, verifica se um fenótipo idêntico já teve sua adaptação calculada. Caso não tenha, aplica o cálculo da a-
        daptação.
        """

        # Carrega o lado, em metros, do elemento de membrana quadrada
        l = self.lado_dos_elementos

        # Chama o algoritmo de identificação da porção útil do gene e construção do fenótipo.
        fenótipo, borda_alcançada, elementos_conectados, nós, me = self._determinar_fenótipo(proj.gene, l)

        if not self._atende_os_requisitos_mínimos(proj, fenótipo, borda_alcançada):
            proj.adaptação = 0
        else:
            # Checa se este fenótipo já teve sua adaptação calculada antes
            if fenótipo.data.tobytes() in self.fenótipos_testados:
                # Recupera a adaptação do cache
                proj.adaptação, proj.f, proj.u, proj.malha = self.fenótipos_testados[fenótipo.data.tobytes()]
                print(f"> Adaptação de {proj.nome} já era conhecida pelo seu fenótipo")
            else:
                # Determina que os tempos de execução de cada etapa da análise por elementos_finitos
                # finitos sejam mensurados cada vez que o nome do Projeto terminar em "1"
                monitorar = proj.nome.endswith("1")

                proj.f, proj.u, proj.malha = self.resolver_para(

                    monitorar=monitorar,
                    malha=Malha(elementos_conectados, nós, me),
                    parâmetros_dos_elementos={"l": l,
                                              "t": self.parâmetros_do_problema["ESPESSURA_DO_ELEMENTO"],
                                              "v": self.parâmetros_do_problema["COEFICIENTE_DE_POYSSON"],
                                              "E": self.parâmetros_do_problema["MÓDULO_DE_YOUNG_DO_MATERIAL"]}

                )

                # Determina as áreas conectadas e desconectadas
                Acon = fenótipo.sum() * (l ** 2)
                Ades = proj.gene.sum() * (l ** 2) - Acon

                # Calcula o deslocamento máximo como a raiz quadrada do maior
                # valor de u_x² + u_y² dentre todos os nós da malha
                n = len(nós)
                Dmax = np.sqrt(np.sum(proj.u.reshape((n, 2)) ** 2, axis=1).max())

                penalização = Dmax - self.Dlim if Dmax > self.Dlim else 0

                if penalização:
                    print(f"> Projeto {proj.nome} penalizado: Dmax - Dlim = {penalização:.3e} metros")

                proj.adaptação = 1 / (Acon + self.e * Ades + self.alfa * penalização)

                print(f"> {proj.nome} conectado à borda. Adaptação: {proj.adaptação}")

                self.fenótipos_testados[fenótipo.data.tobytes()] = proj.adaptação, proj.f, proj.u, proj.malha

        proj.adaptação_testada = True


    def _determinar_fenótipo(self, gene: Gene, l: float
                             ) -> Tuple[Matriz, bool, List[MembranaQuadrada], List[Nó], Matriz]:
        """
        Executa um algoritmo de busca responsável por determinar, para um certo gene cuja expressão fenotípica é dada
        por uma malha de elementos_finitos quadrados de lado l, a maior porção contínua de matéria satisfazendo as res-
        trições do problema, isto é, estar conectada simultaneamente ao ponto de aplicação da força e à borda.

        Também cuida de inicializar a malha correspondente à expressão fenotípica do gene e seus respectivos elementos e
        nós. Embora ter uma função que lide com tantas operações ao mesmo tempo não seja o padrão de programação
        recomendável na maioria dos casos, aqui se justifica pelo ganho em performance.
        """
        gene_útil = np.zeros((self.n, 2*self.n), dtype=bool)

        # Define a posição inicial do algoritmo de busca
        i = int(self.n // 2)
        j = 2*self.n - 1

        elementos = []
        nós = []
        me = []

        # Inicializa estruturas de dados auxiliares que ajudam a manter curso dos índices dos nós e elementos
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
        contexto = (l, gene_útil, elementos, nós, me, índice_na_malha,
                    etiquetas_de_nós_já_construídos, etiquetas_de_elementos_já_construídos)

        # Executa o algoritmo de busca
        while buscando:

            # busca vertical
            partida = (i, j)

            # começar descida
            while descida:

                # consigo descer mais?
                if i != self.n - 1:
                    abaixo = gene[i + 1][j]
                    descida = abaixo
                else:
                    descida = False

                # há ramificações possíveis aqui do lado?
                if j != 2*self.n - 1 and último_movimento != "esquerda":
                    direita = gene[i][j + 1]
                    if direita and not gene_útil[i][j + 1]:
                        possíveis_ramificações.add((i, j + 1, "direita"))

                if j == 0:
                    borda_alcançada = True

                if j != 0 and último_movimento != "direita":
                    esquerda = gene[i][j - 1]
                    if esquerda and not gene_útil[i][j - 1]:
                        possíveis_ramificações.add((i, j - 1, "esquerda"))

                self._adicionar_à_malha_o_elemento_em(i, j, contexto=contexto)
                self._remover_de(possíveis_ramificações, i, j)

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
                if j != 2*self.n - 1 and último_movimento != "esquerda":
                    direita = gene[i][j + 1]
                    if direita and not gene_útil[i][j + 1]:
                        possíveis_ramificações.add((i, j + 1, "direita"))

                if j == 0:
                    borda_alcançada = True

                if j != 0 and último_movimento != "direita":
                    esquerda = gene[i][j - 1]
                    if esquerda and not gene_útil[i][j - 1]:
                        possíveis_ramificações.add((i, j - 1, "esquerda"))

                self._adicionar_à_malha_o_elemento_em(i, j, contexto=contexto)
                self._remover_de(possíveis_ramificações, i, j)

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
    def _adicionar_à_malha_o_elemento_em(i: int, j: int, contexto: tuple) -> None:
        # Recebe o contexto
        (l, gene_útil, elementos, nós, me, índice_na_malha,
         etiquetas_de_nós_já_construídos, etiquetas_de_elementos_já_construídos) = contexto

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
    def _remover_de(possíveis_ramificações: set, i: int, j: int) -> None:
        possíveis_ramificações.discard((i, j, "esquerda"))
        possíveis_ramificações.discard((i, j, "direita"))

    @staticmethod
    def _atende_os_requisitos_mínimos(proj: 'Projeto', fenótipo: Matriz, borda_alcançada: bool) -> bool:
        if not borda_alcançada:
            print(f"> Projeto {proj.nome} desconectado da borda")
        return borda_alcançada

    # Métodos auxiliares da resolução via análise de elementos finitos
    @Monitorador(mensagem="Total de graus de liberdade determinados")
    def determinar_graus_de_liberdade(self, malha: Malha) -> int:
        return 2 * len(malha.nós)

    @Monitorador(mensagem="Matrizes de rigidez local determinadas")
    def calcular_matrizes_de_rigidez_local(self, **parâmetros_do_elemento_base) -> Matriz:
        if self.Ke is None:
            self.Ke = K_base.calcular(parâmetros_do_elemento_base)

        return self.Ke

    @Monitorador(mensagem="Matriz de rigidez global montada")
    def montar_matriz_de_rigidez_geral(self, malha: Malha, Ke: Matriz, graus_de_liberdade: int, método: str) -> Matriz:
        return self._montador_do[método](malha, Ke, graus_de_liberdade)

    @staticmethod
    def montador_expansão(malha: Malha, Ke: Matriz, graus_de_liberdade: int) -> Matriz:
        Kes_expandidos = dict()
        for elemento in malha.elementos:
            Ke_expandido = np.zeros((graus_de_liberdade, graus_de_liberdade))

            índices = np.array([
                [2 * malha.índice_de[n], 2 * malha.índice_de[n] + 1]
                for n in elemento.nós
            ]).flatten()

            for ie in range(len(índices)):
                for je in range(len(índices)):
                    i = índices[ie]
                    j = índices[je]

                    Ke_expandido[i][j] = Ke[ie][je]

            Kes_expandidos[elemento.nós] = Ke_expandido

        return sum(Kes_expandidos.values())

    @staticmethod
    def montador_compacto(malha: Malha, Ke: Matriz, graus_de_liberdade: int) -> Matriz:
        K = np.zeros((graus_de_liberdade, graus_de_liberdade))

        índices = dict()
        for elemento in malha.elementos:
            índices[elemento.nós] = np.array([[2 * malha.índice_de[n], 2 * malha.índice_de[n] + 1]
                                               for n in elemento.nós]).flatten()

        for i in range(8):
            for j in range(8):
                for e in malha.elementos:
                    p = índices[e.nós][i]
                    q = índices[e.nós][j]

                    K[p][q] += Ke[i][j]

        return K

    @staticmethod
    def montador_OptV1(malha: Malha, Ke: Matriz, graus_de_liberdade: int) -> Matriz:
        K = np.zeros((graus_de_liberdade, graus_de_liberdade), dtype=float)

        índices_de_Ke_por_elemento = ((e, i, j) for e in range(malha.ne)
                                      for i in range(8)
                                      for j in range(8))

        for d, (e, i, j) in enumerate(índices_de_Ke_por_elemento):
            K[malha.me[i][e], malha.me[j][e]] += Ke[i][j]

        return K

    @staticmethod
    def montador_OptV2(malha: Malha, Ke: Matriz, graus_de_liberdade: int) -> Matriz:
        K = np.zeros((graus_de_liberdade, graus_de_liberdade), dtype=float)

        índices_de_Ke = ((i, j) for i in range(8) for j in range(8))

        for i, j in índices_de_Ke:
            K[malha.me[i, :], malha.me[j, :]] += Ke[i, j]

        return K

    @Monitorador(mensagem="Condições de contorno incorporadas")
    def incorporar_condições_de_contorno(self,
                                         malha: Malha,
                                         graus_de_liberdade: int,
                                         parâmetros_do_problema: Dict[str, Union[str, int, float]]
                                         ) -> Tuple[Vetor, Vetor, Máscara, Máscara]:
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
