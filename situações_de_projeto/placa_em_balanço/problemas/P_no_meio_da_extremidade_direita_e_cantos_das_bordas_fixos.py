import numpy as np

from situações_de_projeto.placa_em_balanço.problemas.P_no_meio_da_extremidade_direita import PlacaEmBalanço


class PlacaEmBalanço(PlacaEmBalanço):

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

        genes = []
        for k in range(100):
            # Inicia um grafo que representa o preenchimento de cada fatia do espaço de projeto
            grafo = np.random.choice((True, False), (7, 14), p=(0.15, 0.85))

            # Determina os pontos de corte e retorna seus índices i e j
            kis = self._fatiar_intervalo(c=38, t=t, f=7, dividir_ao_meio=True)
            kjs = self._fatiar_intervalo(c=76, t=t, f=14, dividir_ao_meio=False)

            # Caminha aleatoriamente pelo grafo desde o ponto de aplicação da força até uma borda,
            # preenchendo as fatias ao longo da trajetória
            i_chegada=int(np.where(kis == 19)[0])

            trajetória = np.zeros((7, 14), dtype=bool)

            self._caminhada_desde_o_canto_com(i_partida=6, i_chegada=i_chegada, grafo=trajetória, p=0.5)
            self._caminhada_desde_o_canto_com(i_partida=0, i_chegada=i_chegada, grafo=trajetória, p=0.5)

            grafo |= trajetória

            gene = self._formar_gene_a_partir_do(grafo, kis, kjs)

            genes.append(gene)

        return genes

    @staticmethod
    def _caminhada_desde_o_canto_com(i_partida, i_chegada, grafo, p):
        j_partida = 0

        def reta(j):
            return i_partida - j * (i_partida - i_chegada) / 13

        i, j = i_partida, j_partida
        último_movimento = "direita"
        while i != i_chegada or j != 13:
            grafo[i, j] = True

            possíveis_direções = ["baixo", "cima", "direita"]

            # Garante que o algoritmo não volte por onde já passou
            if último_movimento != "direita":
                possíveis_direções.remove("baixo" if "cima" == último_movimento else "cima")
            elif j == 13:
                # Lógica de saída
                if i > i_chegada:
                    i -= 1
                else:
                    i += 1
                último_movimento = "direita"
                continue

            # Garante que o grafo não extrapole os seus limites
            if i == 6 and "baixo" in possíveis_direções:
                possíveis_direções.remove("baixo")
            elif i == 0 and "cima" in possíveis_direções:
                possíveis_direções.remove("cima")

            # Implementa probabilisticamente um atrator para a reta
            if len(possíveis_direções) == 1:
                movimento = "direita"
            elif len(possíveis_direções) == 2:
                p_vertical = (1 - p)/(1 + 3 ** (i - reta(j)))
                if "cima" in possíveis_direções:
                    p_vertical = (1 - p) - p_vertical
                p_vertical += (1 - p)/2

                p_direita = 1 - p_vertical

                movimento = np.random.choice(possíveis_direções, p=(p_vertical, p_direita))
            elif len(possíveis_direções) == 3:
                p_descer = (1 - p) / (1 + 3 ** (i - reta(j)))
                p_subir = (1 - p) - p_descer

                movimento = np.random.choice(possíveis_direções, p=(p_descer, p_subir, p))

            # Move o buscador pelo grafo
            if movimento == "cima":
                i -= 1
            elif movimento == "baixo":
                i += 1
            elif movimento == "direita":
                j += 1

            último_movimento = movimento

        # Adiciona o ponto de chegada ao grafo
        grafo[i, j] = True

    @staticmethod
    def _atende_os_requisitos_mínimos(fenótipo, borda_alcançada):
        if borda_alcançada:
            return fenótipo[0, 0] and fenótipo[-1, 0]
        else:
            return False



