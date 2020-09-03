"""
Fornece a estrutura de classes básicas para executar um algoritmo genético.

CLASSES
-------
Indivíduo
    Objeto que carrega um gene, sua expressão fenotípica para um dado ambiente e sua adaptação para um dado problema.
Ambiente
    Framework de classe de objetos que agregam Indivíduos e definem sobre eles operadores genéticos.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, List
from abc import ABC, abstractmethod

import numpy as np


class Ambiente(ABC):
    """
    Framework de classe de objetos que agregam Indivíduos e definem sobre eles operadores genéticos.

    Um Ambiente implementa condições específicas de teste e combinação de indivíduos descrita pelos seus métodos. É re-
    presentado pelo ranking de adaptação dos indivíduos da população corrente e por uma lista das suas últimas mutações.

    ATRIBUTOS (da classe)
    ---------------------
    genes_testados: dict -- Cache de valores de adaptação de genes já testados

    ATRIBUTOS (dos objetos)
    -----------------------
    probabilidade_de_mutar: float                -- Chance base de um bit de gene virar em decorrência de uma mutação
    população             : List[Indivíduo]      -- Carrega os indivíduos da geração corrente
    gerações              : List[List[Indivíduo] -- Carrega históricos de cada geração. Limpa-se e se sumariza durante a
                                                    execução do código com soluções específicas.
    mutações              : List[str]            -- Histórico de mutações mais recentes.
    n_da_geração          : int                  -- Número identificador da geração corrente
    n_de_indivíduos       : int                  -- Quantidade de indivíduos da geração corrente

    MÉTODOS CONCRETOS
    -----------------
    avançar_gerações(self, n: int) -> self
        Faz a população corrente avançar n gerações.
    próxima_geração(self) -> self
        Faz a população corrente avançar uma geração.
    seleção_natural(self) -> List[Indivíduo]
        Seleciona os indivíduos com melhores genes.
    reprodução(self, indivíduos_selecionados: List[Indivíduo]) -> List[Indivíduo]
        Determina em quais indivíduos aplicar o operador de crossover para gerar novos indivíduos filhos.

    MÉTODOS ABSTRATOS
    -----------------
    geração_0(self) -> List[Indivíduo]
        Construirá a geração inicial da população do ambiente.
    testar_adaptação(self, indivíduo: Indivíduo) -> None
        Testará a adaptação do indivíduo, modificando seus atributos.
    crossover(self, pai1: Indivíduo, pai2: Indivíduo, i: int) -> Indivíduo
        Gerará um novo indivíduo a partir do cruzamento dos genes de seus pais.
    mutação(self, geração: List[Indivíduo]) -> None
        Modificará a nova geração de acordo com as regras estabelecidas pelo ambiente.

    MÉTODOS AUXILIARES
    ------------------
    _conseguir_adaptação(self, ind: Indivíduo) -> None
        Chamado por seleção_natural e reprodução.
        Checa se o gene do indivíduo já teve sua adaptação testada e armazena os valores já calculados.
    """

    genes_testados = dict()

    def __init__(self, indivíduos: Optional[List['Indivíduo']] = None, probabilidade_de_mutar: float = 0.0):
        if indivíduos is None:
            indivíduos = self.geração_0()

        self.gerações               = [indivíduos]
        self.população              = indivíduos
        self.n_da_geração           = 0
        self.n_de_indivíduos        = len(indivíduos)
        self.probabilidade_de_mutar = probabilidade_de_mutar

    @abstractmethod
    def geração_0(self) -> List['Indivíduo']:
        """Construirá a geração inicial da população do ambiente."""

    def avançar_gerações(self, n: int) -> "Ambiente":
        """Faz a população corrente avançar n gerações"""
        for _ in range(n):
            self.próxima_geração()

        return self

    def próxima_geração(self) -> "Ambiente":
        """Faz a população corrente avançar uma geração"""
        indivíduos_selecionados = self.seleção_natural()
        nova_geração            = self.reprodução(indivíduos_selecionados)

        self.mutação(nova_geração)

        novos_indivíduos = indivíduos_selecionados + nova_geração
        novos_indivíduos.sort(reverse=True)

        self.gerações.append(novos_indivíduos)
        self.população = novos_indivíduos

        self.n_da_geração += 1

        return self

    def seleção_natural(self) -> List['Indivíduo']:
        """
        Seleciona os indivíduos com melhores genes.

        Testa os indivíduos cuja adaptação ainda não foi calculada e retorna a metade mais adaptada numa lista.

        Retorna
        -------
        indivíduos_selecionados: List[Projeto] -- Vencedores da seleção natural
        """

        # Testa os indivíduos ainda não adaptados da população
        for ind in self.população:
            if not ind.adaptação_testada:
                self._conseguir_adaptação(ind)

        # Ordena os indivíduos por adaptação decrescente e filtra a metade superior
        indivíduos_selecionados = sorted(self.população, reverse=True)[:self.n_de_indivíduos // 2]

        return indivíduos_selecionados

    def _conseguir_adaptação(self, ind: 'Indivíduo') -> 'Indivíduo':
        """Checa se o gene do indivíduo já teve sua adaptação testada e armazena os valores já calculados."""
        if ind.id in self.genes_testados.keys():
            ind.adaptação = self.genes_testados[ind.id]
        else:
            self.testar_adaptação(ind)
            self.genes_testados[ind.id] = ind.adaptação

        return ind

    @abstractmethod
    def testar_adaptação(self, indivíduo: 'Indivíduo') -> None:
        """Testará a adaptação do indivíduo, modificando seus atributos."""

    def reprodução(self, indivíduos_selecionados: List['Indivíduo']) -> List['Indivíduo']:
        """Determina em quais indivíduos aplicar o operador de crossover para gerar novos indivíduos filhos.

        Define probabilidades de reproduzir para cada indivíduo proporcionalmente às suas adaptações. Seleciona dentre
        elas aleatoriamente e executa o operador de crossover tantas vezes quanto seja necessário para gerar a quanti-
        dade de indivíduos filhos desejada.

        ARGUMENTOS
        ----------
        inds  : List[Indivíduo] -- Lista de indivíduos selecionados para sobrevivência e reprodução.

        RETORNA
        -------
        filhos: List[Indivíduo] -- Resultado da aplicação sucessiva do operador de crossover
        """

        filhos     = []
        adaptações = np.array([i.adaptação for i in indivíduos_selecionados])

        for k in range(self.n_de_indivíduos - len(indivíduos_selecionados)):

            probabilidades = adaptações/(adaptações.sum())

            # Escolhe dois indivíduos distintos como pais de acordo com suas probabilidades de reprodução
            pais = np.random.choice(indivíduos_selecionados, size=2, replace=False, p=probabilidades)

            ind_filho = self.crossover(pais[0], pais[1], k + 1)
            self._conseguir_adaptação(ind_filho)
            filhos.append(ind_filho)

        return filhos

    @abstractmethod
    def crossover(self, pai1: 'Indivíduo', pai2: 'Indivíduo', i: int) -> 'Indivíduo':
        """ Gerará um novo indivíduo a partir do cruzamento dos genes de seus pais."""

    @abstractmethod
    def mutação(self, geração: List['Indivíduo']) -> None:
        """Modificará a nova geração de acordo com as regras estabelecidas pelo ambiente."""

    def __repr__(self):
        return f"Geração {self.n_da_geração} de População de {self.n_de_indivíduos} indivíduos: {self.população!s}"

    def __str__(self):
        return (f"População de {self.n_de_indivíduos} indivíduos em sua geração {self.n_da_geração}:\n"
                + (self.n_de_indivíduos * "> {}\n").format(*self.população))


@dataclass(order=True)
class Indivíduo:
    """Objeto que carrega um gene, sua expressão fenotípica para um dado ambiente e sua adaptação para um dado problema.

    FUNCIONALIDADES
    ---------------
    1. Indivíduos podem ser comparados de acordo com sua adaptação:
       >>> Indivíduo(gene="01001010", nome="ind1", adaptação=0) < Indivíduo(gene="10010001", nome="ind2", adaptação=1)
       True
    """
    gene: Any = field(compare=False)
    nome: str = field(compare=False)
    adaptação: float = 0.0
    adaptação_testada: bool = field(default=False, compare=False)

    def __post_init__(self):
        self.id = self.gene

    def __str__(self):
        return f"{self.nome}: {self.adaptação}"
