"""
Fornece a estrutura de classes básicas para executar um algoritmo genético.

CLASSES
-------
Indivíduo
    Classe de objetos que carregam genes, elementos_finitos fenotípicos e valores de adaptação.
Ambiente
    Classe de objetos que agregam Indivíduos e definem sobre eles operadores genéticos.
"""
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Ambiente(ABC):
    """
    Classe de objetos que agregam Indivíduos e definem sobre eles operadores genéticos.

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
    """
    
    genes_testados = dict()
    
    def __init__(self, indivíduos=None, probabilidade_de_mutar=0):
        if indivíduos is None:
            indivíduos = self.geração_0()
            
        self.probabilidade_de_mutar = probabilidade_de_mutar
        self.população              = indivíduos
        self.gerações               = [indivíduos]
        self.mutações               = []
        self.n_da_geração           = 0
        self.n_de_indivíduos        = len(indivíduos)

    def avançar_gerações(self, n):
        for _ in range(n):
            self.próxima_geração()

        return self
        
    def próxima_geração(self):
        self.n_da_geração += 1
        
        indivíduos_selecionados = self.seleção_natural()
        nova_geração            = self.reprodução(indivíduos_selecionados)

        self.mutação(nova_geração)

        novos_indivíduos = indivíduos_selecionados + nova_geração
        novos_indivíduos.sort(reverse=True)
        
        self.gerações.append(novos_indivíduos)
        self.população = novos_indivíduos
        
        return self
        
    def seleção_natural(self):
        """
        Seleciona os indivíduos com melhores genes.

        Testa os indivíduos cuja adaptação ainda não foi calculada e retorna a metade mais adaptada numa lista

        Retorna
        -------
        indivíduos_selecionados: List[Projeto] -- Vencedores da seleção natural
        """

        # Testa os indivíduos ainda não adaptados da população
        for ind in self.população:
            if not ind.adaptação_testada:
                self.conseguir_adaptação(ind)

        # Ordena os indivíduos por adaptação decrescente e filtra a metade superior
        indivíduos_selecionados = sorted(self.população, reverse=True)[:self.n_de_indivíduos // 2]

        return indivíduos_selecionados

    def conseguir_adaptação(self, ind):
        """Checa se o gene do indivíduo já teve sua adaptação testada e armazena os valores já calculados."""
        if ind.id in self.genes_testados.keys():
            ind.adaptação = self.genes_testados[ind.id]
        else:
            self.testar_adaptação(ind)
            self.genes_testados[ind.id] = ind.adaptação
        
    def reprodução(self, inds):
        """
        Determina em quais indivíduos aplicar o operador de crossover para gerar novos indivíduos filhos.

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
        adaptações = np.array([i.adaptação for i in inds])
        
        for k in range(self.n_de_indivíduos - len(inds)):

            probabilidades = adaptações/(adaptações.sum())

            # Escolhe dois indivíduos distintos como pais de acordo com suas probabilidades de reprodução
            pais = np.random.choice(inds, size=2, replace=False, p=probabilidades)
                
            ind_filho = self.crossover(pais[0], pais[1], k + 1)
            self.conseguir_adaptação(ind_filho)
            filhos.append(ind_filho)
            
        return filhos
            
    @abstractmethod
    def crossover(self, ind1, ind2, i):
        pass
    
    @abstractmethod
    def mutação(self, geração):
        pass

    @abstractmethod
    def testar_adaptação(self, indivíduo):
        pass

    @abstractmethod
    def geração_0(self):
        pass
    
    def __repr__(self):
        return "Geração {} de População de {} indivíduos: {}".format(self.n_da_geração, self.n_de_indivíduos, self.população)
    
    def __str__(self):
        return ("População de {} indivíduos em sua geração {}:\n".format(self.n_de_indivíduos, self.n_da_geração)
                + (self.n_de_indivíduos * "> {}\n").format(*self.população)
                + "---------Mutações---------\n"
                + "\n".join(self.mutações))


@dataclass(order=True)
class Indivíduo:
    gene: Any = field(compare=False)
    nome: str = field(compare=False)
    adaptação: float = 0.0
    adaptação_testada: bool = field(default=False, compare=False)

    def __post_init__(self):
        self.id = self.gene

    def __str__(self):
        return f"{self.nome}: {self.adaptação}"
