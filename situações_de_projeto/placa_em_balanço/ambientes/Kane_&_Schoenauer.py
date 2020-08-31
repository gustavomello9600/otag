from dataclasses import dataclass, field
from typing import Optional
from random import choice

import numpy as np
from more_itertools import grouper

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

    def __init__(self, problema, indivíduos=None, n_de_indivíduos=125, probabilidade_de_mutar=0.01/100):
        self.problema = problema
        self.n_de_indivíduos = n_de_indivíduos
        self.índice_de_convergência = 0

        super().__init__(indivíduos=indivíduos, probabilidade_de_mutar=probabilidade_de_mutar)

    def geração_0(self):
        return [Projeto(gene, nome=f"Proj_{i + 1}")
                for i, gene in enumerate(self.problema.geração_0(self.n_de_indivíduos))]

    def próxima_geração(self):
        self.problema.alfa = self.problema.alfa_0 * (1.01 ** self.n_da_geração)

        print(f"\nGeração {self.n_da_geração}"
              f"\n-----------")

        self.seleção_natural()
        self.reprodução()
        self.mutação()

        self.n_da_geração += 1

        return self

    def seleção_natural(self):
        # Testa os indivíduos ainda não adaptados da população
        for ind in self.população:
            if not ind.adaptação_testada:
                self._conseguir_adaptação(ind)

        self.população.sort(reverse=True)

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

    def reprodução(self, população=None):
        adaptações = np.array([i.adaptação for i in self.população])
        probabilidades = adaptações / (adaptações.sum())

        self.população = [Projeto(p.gene.copy(), p.nome, u=p.u, f=p.f, malha=p.malha) for p in

            np.random.choice(self.população, self.n_de_indivíduos, p=probabilidades, replace=True)

        ]

        for par_de_projetos in grouper(self.população, 2):
            if np.random.random() < 0.6:
                proj1, proj2 = par_de_projetos
                if proj2 is not None:
                    self.crossover(proj1, proj2)

    def crossover(self, p1, p2, índice=None):
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
        gene_de_p1_antes_do_crossover = p1.gene.copy()
        gene_de_p2_antes_do_crossover = p2.gene.copy()
        for bloco in blocos:
            ibc, ibb = bloco[0]
            jbe, jbd = bloco[1]

            p1.gene[ibc:ibb, jbe:jbd] = gene_de_p2_antes_do_crossover[ibc:ibb, jbe:jbd]
            p2.gene[ibc:ibb, jbe:jbd] = gene_de_p1_antes_do_crossover[ibc:ibb, jbe:jbd]

    def mutação(self, população=None):
        mapa_de_convergência, índice_de_convergência = self._calcular_índice_de_convergência()

        self._mutação_baseada_na_população(mapa_de_convergência)

        if índice_de_convergência > 0.75:
            self._mutação_baseada_na_topologia()

        self.índice_de_convergência = índice_de_convergência

    def _calcular_índice_de_convergência(self):
        genes = [proj.gene for proj in self.população]
        mapa_de_convergência   = np.mean(genes, axis=0)
        índice_de_convergência = sum(self._mconv(mapa_de_convergência).flat) / len(mapa_de_convergência.flat)
        return mapa_de_convergência, índice_de_convergência

    @staticmethod
    @np.vectorize
    def _mconv(V):
        return 4 * (V ** 2) - 4 * V + 1

    def _mutação_baseada_na_população(self, mapa_de_convergência):
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
        Médias = mapa_de_convergência
        Médias_2 = Médias ** 2

        for ind in self.população:
            # Obtém a propabilidade de mutar de cada bit
            probabilidade_de_mutar = (self.probabilidade_de_mutar
                                      + 99 * self.probabilidade_de_mutar * Médias_2
                                      + 99 * self.probabilidade_de_mutar * ind.gene * (1 - 2 * Médias))

            # Sorteia os casos em que há mutação
            mutações = probabilidade_de_mutar > np.random.random((38, 76))

            # Vira os bits que resultaram em mutações
            ind.gene[mutações] = ~ind.gene[mutações]

    def _mutação_baseada_na_topologia(self):
        pm = 50 * self.probabilidade_de_mutar

        for ind in self.população:
            gene = ind.gene
            bordas = ((gene ^ np.roll(gene, 1))
                      | (gene ^ np.roll(gene, -1))
                      | (gene ^ np.roll(gene, 1, axis=0))
                      | (gene ^ np.roll(gene, -1, axis=0)))

            aumentar_bordas = True if 0.5 > np.random.random() else False
            if aumentar_bordas:
                bordas_sujeitas_a_mutação = gene & bordas
            else:
                bordas_sujeitas_a_mutação = ~gene & bordas

            bits_virados = (bordas_sujeitas_a_mutação
                            & np.random.choice((True, False), bordas_sujeitas_a_mutação.shape, p=(pm, 1 - pm)))

            gene[bits_virados] = ~gene[bits_virados]

    def finalizar(self):
        print("\nGarantindo malha do projeto mais bem adaptado"
              "\n---------------------------------------------")
        self.população.sort(reverse=True)
        self.testar_adaptação(self.população[0])


@dataclass(order=True)
class Projeto(Indivíduo):
    """Classe que carrega as propriedades de cada projeto."""

    u: Optional['ArrayLike'] = field(default=None, compare=False)
    f: Optional['ArrayLike'] = field(default=None, compare=False)
    malha: Optional['Malha'] = field(default=None, compare=False)

    def __post_init__(self):
        self.id = self.gene.data.tobytes()
