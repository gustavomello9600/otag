"""
Fornece a estrutura de classes básicas para executar um algoritmo genético.

CLASSES
-------
Indivíduo
    Classe de objetos que carregam genes, elementos fenotípicos e valores de adaptação.
População
    Classe de objetos que agregam Indivíduos e definem sobre eles operadores genéticos.
"""


import numpy as np


class Indivíduo:
    """
    Classe de objetos que carregam genes, elementos fenotípicos e valores de adaptação.

    São definidas operações de comparação baseadas nos valores de adaptação e um método de representação que traz
    seus nomes, suas adaptações e representações curtas de seus genes.

    ATRIBUTOS
    ---------
    nome             : str    -- Identificador do indivíduo. Normalmente o situa numa população.
    adaptação        : Number -- Valor que qualifica seu gene em um determinado ambiente/problema.
    adaptação_testada: bool   -- Verdadeiro se já foi determinada e falso caso contrário.
    """
    
    def __init__(self, gene, nome=None):
        self.gene = gene
        if nome is None:
            self.nome = self.gerar_nome_aleatório()
        else:
            self.nome = nome
            
        self.adaptação         = 0
        self.adaptação_testada = False
        self.id_do_gene        = self.gerar_id_do_gene()

    def gerar_id_do_gene(self):
        pass
            
    def __repr__(self):
        return f"{self.nome}: {self.adaptação:.2f} ({self.gene.__repr__()[:15]}...)"
    
    def __gt__(self, other):
        return self.adaptação  > other.adaptação
    
    def __lt__(self, other):
        return self.adaptação  < other.adaptação
    
    def __eq__(self, other):
        return self.adaptação == other.adaptação


class População:
    """
    Classe de objetos que agregam Indivíduos e definem sobre eles operadores genéticos.

    Uma População implementa condições específicas de teste e combinação de indivíduos descrita pelos seus métodos.
    É representada pelo rankind de adaptação de seus indivíduos e por uma lista das suas últimas mutações.

    ATRIBUTOS (da classe)
    ---------------------
    genes_testados        : dict                 -- Cache de valores de adaptação de genes já testados

    ATRIBUTOS (dos objetos)
    -----------------------
    probabilidade_de_mutar: float                -- Chance base de um bit de gene virar em decorrência de uma mutação
    indivíduos            : List[Indivíduo]      -- Carrega os indivíduos da geração corrente
    gerações              : List[List[Indivíduo] -- Carrega históricos de cada geração. Limpa-se e se sumariza durante a
                                                    execução do código com soluções específicas.
    mutações              : List[str]            -- Histórico de mutações mais recentes.
    n_da_geração          : int                  -- Número identificador da geração corrente
    n_de_indivíduos       : int                  -- Quantidade de indivíduos da geração corrente
    """
    
    genes_testados = dict()
    
    def __init__(self, indivíduos=None, probabilidade_de_mutar=0.01/100):
        if indivíduos is None:
            indivíduos = self.geração_0()
            
        self.probabilidade_de_mutar = probabilidade_de_mutar
        self.indivíduos             = indivíduos
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
        self.indivíduos = novos_indivíduos
        
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
        for ind in self.indivíduos:
            if not ind.adaptação_testada:
                self.conseguir_adaptação(ind)

        # Ordena os indivíduos por adaptação decrescente e filtra a metade superior
        indivíduos_selecionados = sorted(self.indivíduos, reverse=True)[:self.n_de_indivíduos // 2]

        return indivíduos_selecionados

    def conseguir_adaptação(self, ind):
        """Checa se o gene do indivíduo já teve sua adaptação testada e armazena os valores já calculados."""
        if ind.id_do_gene in self.genes_testados.keys():
            ind.adaptação = self.genes_testados[ind.id_do_gene]
        else:
            self.testar_adaptação(ind)
            self.genes_testados[ind.id_do_gene] = ind.adaptação
        
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
            
    # Sobrescrever
    def crossover(self, ind1, ind2, i):
        pass
    
    # Sobrescrever
    def mutação(self, geração):
        pass

    # Sobrescrever
    def testar_adaptação(self, indivíduo):
        pass

    # Sobrescrever
    def geração_0(self):
        pass
    
    def __repr__(self):
        return "Geração {} de População de {} indivíduos: {}".format(self.n_da_geração, self.n_de_indivíduos, self.indivíduos)
    
    def __str__(self):
        return ("População de {} indivíduos em sua geração {}:\n".format(self.n_de_indivíduos, self.n_da_geração)
                + (self.n_de_indivíduos * "> {}\n").format(*self.indivíduos)
                + "---------Mutações---------\n"
                + "\n".join(self.mutações))
