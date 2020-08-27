from random import choice
from dataclasses import dataclass

import numpy as np

from suporte.algoritmo_genético import Ambiente, Indivíduo


class AmbienteDeProjeto(Ambiente):
    """
    Classe de objetos que agregam os indivíduos de uma população de projetos e os manipulam de acordo
    com operadores genéticos específicos ao problema. Herda seus atributos da classe Ambiente definida em
    suporte.algoritmo_genético.py e tem a maior parte dos seus métodos sobrescritos aqui.

    Métodos
    -------
    OPERADORES GENÉTICOS
    próxima_geração() -> None
        Executa todos os passos necessários para avançar uma geração como implementado pela classe População
        no módulo suporte.algoritmo_genético. Adicionalmente, atualiza o valor de alfa.
    crossover(p1: Projeto, p2: Projeto, índice: int) -> Projeto
        Gera um indivíduo filho a partir do cruzamento de dois indivíduos pais.
    mutação(nova_geração: List[Projeto]) -> None
        Vira alguns bits dos genes dos indivíduos da próxima geração da população de acordo com uma
        probabilidade de mutar definida na construção da instância.
    testar_adaptação(ind: Projeto) -> None
        Testa a adaptação do indivíduo utilizando a modelagem e as condições de contorno do problema.
    """

    def __init__(self, problema, indivíduos=None, probabilidade_de_mutar=0.01/100):
        self.problema = problema
        super().__init__(indivíduos=indivíduos, probabilidade_de_mutar=probabilidade_de_mutar)

    def geração_0(self):
        return [Projeto(gene, nome=f"G0_{i + 1}") for i, gene in enumerate(self.problema.geração_0())]

    def próxima_geração(self):
        self.problema.alfa = self.problema.alfa_0 * (1.01 ** self.n_da_geração)
        super().próxima_geração()

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
        gene_novo = p1.gene.copy()

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
        Médias = sum([ind.gene for ind in self.população]) / self.n_de_indivíduos
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

        self.problema.testar_adaptação(ind)


@dataclass(order=True)
class Projeto(Indivíduo):
    """Classe que carrega as propriedades de cada projeto."""

    def __post_init__(self):
        self.id = self.gene.data.tobytes()
