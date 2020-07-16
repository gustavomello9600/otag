#!/usr/bin/env python
# coding: utf-8

import uuid
import numpy as np
from copy import copy

class Indivíduo:
    
    def __init__(self, gene, nome=""):
        self.gene = gene
        if nome == "":
            self.nome = str(uuid.uuid4())[:4]
        else:
            self.nome = nome
            
        self.adaptação         = 0
        self.adaptação_testada = False
        self.gene_id = None

    def id_do_gene(self):
        if self.gene_id is None:
            self.gene_id = self.gerar_id_do_gene()

        return self.gene_id

    # Sobrescrever
    def gerar_id_do_gene(self):
        return self.gene
            
    def __repr__(self):
        return "{}: {:.2f} ({}...)".format(self.nome, self.adaptação, (self.gene).__repr__()[:15])
    
    def __gt__(self, other):
        return self.adaptação  > other.adaptação
    
    def __lt__(self, other):
        return self.adaptação  < other.adaptação
    
    def __eq__(self, other):
        return self.adaptação == other.adaptação
    
    #Sobrescrever
    def construir_fenótipo():
        self.fenótipo = None


class População:
    
    genes_testados = dict()
    
    def __init__(self, indivíduos=None, pm=0.01/100):
        if indivíduos is None:
            indivíduos = self.geração_0()
            
        self.geração    = 0
        self.indivíduos = indivíduos
        self.gerações   = [indivíduos]
        self.n          = len(indivíduos)
        self.mutações   = []
        self.pm         = pm

    def avançar_gerações(self, n):
        for _ in range(n):
            self.próxima_geração()

        return self
        
    def próxima_geração(self):
        self.geração += 1
        
        indivíduos_selecionados = self.seleção_natural()
        nova_geração            = self.reprodução(indivíduos_selecionados)

        self.mutação(nova_geração)

        novos_indivíduos = indivíduos_selecionados + nova_geração
        novos_indivíduos.sort(reverse=True)
        
        self.gerações.append(novos_indivíduos)
        self.indivíduos    = novos_indivíduos
        
        return self
        
    def seleção_natural(self):
        for ind in self.indivíduos:
            if not ind.adaptação_testada:
                self.conseguir_adaptação(ind)
                
        return sorted(self.indivíduos, reverse=True)[:self.n//2]
        
    def conseguir_adaptação(self, ind):
        if ind.id_do_gene() in self.genes_testados.keys():
            ind.adaptação = self.genes_testados[ind.id_do_gene()]
        else:
            self.testar_adaptação(ind)
            self.genes_testados[ind.id_do_gene()] = ind.adaptação
        
    def reprodução(self, inds):
        filhos = []
        
        pesos = np.array([i.adaptação for i in inds])
        
        for k in range(self.n - len(inds)):
            pais = np.random.choice(inds, size=2, replace=False, p=pesos/(pesos.sum()))
                
            ind_filho = self.crossover(pais[0], pais[1], k + 1)
            self.conseguir_adaptação(ind_filho)
            filhos.append(ind_filho)
            
        return filhos
            
    #Sobrescrever
    def crossover(self, ind1, ind2, i):
        k1, k2 = np.random.randint(1, 8, (2,))
        
        while k1 == k2:
            k2 = np.random.randint(1, 8, (1,))
            
        kmax, kmin = int(max(k1, k2)), int(min(k1, k2))
        
        return Indivíduo(ind1.gene[:kmin] + ind2.gene[kmin:kmax] + ind1.gene[kmax:],
                         nome="G{}_{}".format(self.geração, i))
    
    #Sobrescrever
    def mutação(self, g):
        for ind in g:
            mutated = False
            for i, b in enumerate(ind.gene):
                p = np.random.random()
                if p < self.pm:
                    mutated = True
                    ind.gene = ind.gene[:i] + ("0" if ind.gene[i] == "1" else "1") + ind.gene[i + 1:]
            if mutated:
                ind.adaptação_testada = False
                self.mutações.append("Indivíduo {} indo para a geração {}".format(
                                     ind.nome, self.geração - 1))

    #Sobrescrever
    def testar_adaptação(self, ind):
        ind.adaptação = sum([int(n)*(2**i) for i, n in enumerate(ind.gene)])
    
    def geração_0(self):
        return [Indivíduo("".join([str(b) for b in np.random.randint(0, 2, (8,))]), nome="G0_{}".format(i))
                                                                                          for i in range(10)]
    
    def __repr__(self):
        return "Geração {} de População de {} indivíduos: {}".format(self.geração,self.n, self.indivíduos)
    
    def __str__(self):
        return ("População de {} indivíduos em sua geração {}:\n".format(self.n, self.geração)
                + (self.n *"> {}\n").format(*self.indivíduos)
                + "---------Mutações---------\n"
                + "\n".join(self.mutações))


if __name__ == "__main__":
    np.random.seed(0)
    
    pop = População()
    pop.avançar_gerações(10)

    from matplotlib import pyplot as plt

    X = np.arange(11)
    Y = [np.mean(adg) for adg in [[ind.adaptação for ind in gen] for gen in pop.gerações]]

    plt.plot(X, Y)

    plt.show()
