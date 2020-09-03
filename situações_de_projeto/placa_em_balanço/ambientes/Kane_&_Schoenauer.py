from dataclasses import dataclass, field
from typing import Optional, List
from random import choice
import multiprocessing

import numpy as np
from more_itertools import grouper

from suporte.algoritmo_genético import Ambiente, Indivíduo


class AmbienteDeProjeto(Ambiente):

    def __init__(self,
                 problema: 'Problema',
                 indivíduos: Optional[List['Projeto']] = None,
                 n_de_indivíduos: int = 125,
                 probabilidade_de_mutar: float = 0.01/100,
                 paralelização: bool = False):

        self.problema = problema
        self.paralelizado = paralelização
        self.n_de_indivíduos = n_de_indivíduos
        self.índice_de_convergência = 0

        super().__init__(indivíduos=indivíduos, probabilidade_de_mutar=probabilidade_de_mutar)

    def geração_0(self):
        return [Projeto(gene, nome=f"Proj_{i + 1}")
                for i, gene in enumerate(self.problema.geração_0(self.n_de_indivíduos))]

    def próxima_geração(self):
        self.problema.alfa = self.problema.alfa_0 * (1.01 ** self.n_da_geração)

        print(f"\nGeração {self.n_da_geração}"
              f"\n===========")

        self.seleção_natural()
        self.reprodução()
        self.mutação()

        self.n_da_geração += 1

        return self

    def seleção_natural(self):
        """Testa os indivíduos ainda não adaptados da população"""
        if self.paralelizado:
            self.seleção_natural_em_paralelo()
        else:
            self.seleção_natural_em_série()

        self.população.sort(reverse=True)

    def seleção_natural_em_paralelo(self):
        """ Distribui entre os núcleos do processador o trabalho de conseguir a adaptação de cada projeto."""
        with multiprocessing.Pool() as fila_de_processamento:
            self.população = fila_de_processamento.map(self._conseguir_adaptação, self.população)

    def seleção_natural_em_série(self):
        for proj in self.população:
            self._conseguir_adaptação(proj)

    def testar_adaptação(self, ind):
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
        mapa_de_convergência, self.índice_de_convergência = self._calcular_índice_de_convergência()
        print(f"\nMutações (Índice de Convergência: {100*self.índice_de_convergência:.2f}%)"
              f"\n--------")

        self._mutação_baseada_na_população(mapa_de_convergência)
        print(f"> Aplicado operador de convergência em toda a população")

        if self.índice_de_convergência > 0.75:
            self._mutação_baseada_na_topologia()
            print(f"> Aplicado operador de mutação nas bordas em toda a população")

    def _calcular_índice_de_convergência(self):
        genes = [proj.gene for proj in self.população]
        mapa_de_convergência = np.mean(genes, axis=0)
        índice_de_convergência = self._parábola(mapa_de_convergência).flatten().mean()

        return mapa_de_convergência, índice_de_convergência

    @staticmethod
    @np.vectorize
    def _parábola(V):
        return 4 * (V ** 2) - 4 * V + 1

    def _mutação_baseada_na_população(self, mapa_de_convergência):
        # Obtém a média, e a média ao quadrado, de cada bit na população
        Médias = mapa_de_convergência
        Médias_2 = Médias ** 2

        # Passível de paralelização
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
        pm = 10 * self.probabilidade_de_mutar

        # Passível de paralelização
        for ind in self.população:
            gene = ind.gene
            bordas = ((gene ^ np.roll(gene, 1))
                      | (gene ^ np.roll(gene, -1))
                      | (gene ^ np.roll(gene, 1, axis=0))
                      | (gene ^ np.roll(gene, -1, axis=0)))

            aumentar_bordas = True if 0.5 > np.random.random() else False
            if aumentar_bordas:
                bordas_sujeitas_a_mutação = ~gene & bordas
            else:
                bordas_sujeitas_a_mutação = gene & bordas

            bits_virados = (bordas_sujeitas_a_mutação
                            & np.random.choice((True, False),
                                               bordas_sujeitas_a_mutação.shape,
                                               p=(pm, 1 - pm)))

            gene[bits_virados] = ~gene[bits_virados]

    def finalizar(self):
        print("\nExecução Final"
              "\n--------------")
        self.seleção_natural()


@dataclass(order=True)
class Projeto(Indivíduo):
    """Classe que carrega as propriedades de cada projeto."""

    u: Optional['ArrayLike'] = field(default=None, compare=False)
    f: Optional['ArrayLike'] = field(default=None, compare=False)
    malha: Optional['Malha'] = field(default=None, compare=False)

    def __post_init__(self):
        self.id = self.gene.data.tobytes()
