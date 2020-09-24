from copy import deepcopy
from itertools import cycle
from unittest.mock import Mock

import pytest

from situações_de_projeto.placa_em_balanço.ambientes.Kane_e_Schoenauer import *


@pytest.fixture
def adaptações():
    alternativas = cycle([0, 0.2, 0.4, 0.6, 0.8, 1])
    return alternativas


@pytest.fixture
def maquete_de_problema(adaptações):
    problema = Mock()
    problema.alfa_0 = 1

    def testar_projeto_fake(projeto):
        projeto.adaptação = next(adaptações)
        projeto.adaptação_testada = True
        projeto.u, projeto.f, projeto.malha = np.empty((3, 1)), np.empty((3, 1)), np.empty((6, 6))

    problema.testar_adaptação = testar_projeto_fake

    genes_0 = [np.zeros((6, 6), dtype=bool) for _ in range(4)]
    genes_0[0][0:2, 0:2] = True
    genes_0[1][0:2, 4:6] = True
    genes_0[2][4:6, 4:6] = True
    genes_0[3][4:6, 0:2] = True

    problema.geração_0.return_value = genes_0
    problema.genes_0 = genes_0

    return problema


@pytest.fixture
def ambiente_de_teste(maquete_de_problema):
    return AmbienteDeProjeto(maquete_de_problema, n_de_indivíduos=4)


def teste_inicializar_AmbienteDeProjeto(maquete_de_problema, ambiente_de_teste):
    ambiente = ambiente_de_teste

    assert ambiente.n_de_indivíduos == 4
    assert ambiente.problema == maquete_de_problema
    assert ambiente.população == [Projeto(gene, nome=f"Proj_{i + 1}") 
                                  for i, gene in enumerate(maquete_de_problema.genes_0)]


def teste_próxima_geração(ambiente_de_teste, maquete_de_problema):
    random.seed(0)
    np.random.seed(0)

    ambiente_de_teste.próxima_geração()

    assert np.all(ambiente_de_teste.população[0].gene == maquete_de_problema.genes_0[2])


def teste_seleção_natural(ambiente_de_teste):
    ambiente_de_teste.seleção_natural()

    assert ambiente_de_teste.população[0].adaptação == 0.6
    assert ambiente_de_teste.população[1].adaptação == 0.4
    assert ambiente_de_teste.população[2].adaptação == 0.2
    assert ambiente_de_teste.população[3].adaptação == 0


def teste_reprodução(ambiente_de_teste, monkeypatch):
    ambiente_de_teste.seleção_natural()
    população_antes_da_reprodução = deepcopy(ambiente_de_teste.população)
    população_esperada = população_antes_da_reprodução[:3] + população_antes_da_reprodução[:1]

    crossover_chamado = False
    números_sorteados = cycle([0.2, 0.8])

    def retornar_na_mesma_ordem_exceto_o_último(população, *args, **kwargs):
        return população[:3] + população[:1]

    def retornar_número_sorteado_pré_definido(*args, **kwargs):
        return next(números_sorteados)

    def crossover_falso(*args, **kwargs):
        nonlocal crossover_chamado
        if crossover_chamado:
            raise ValueError("Crossover já havia sido chamado uma vez e foi chamado de novo.")
        crossover_chamado = True

    with monkeypatch.context() as m:
        m.setattr(np.random, "choice", retornar_na_mesma_ordem_exceto_o_último)
        m.setattr(np.random, "random", retornar_número_sorteado_pré_definido)
        m.setattr(ambiente_de_teste, "crossover", crossover_falso)
        ambiente_de_teste.reprodução()

    assert crossover_chamado

    def comparação_por_gene(projeto, outro):
        return np.all(projeto.gene == outro.gene)

    with monkeypatch.context() as m:
        m.setattr(Projeto, "__eq__", comparação_por_gene)
        assert ambiente_de_teste.população == população_esperada


def teste_crossover(ambiente_de_teste, monkeypatch):
    proj_1, proj_2 = ambiente_de_teste.população[:2]

    cortes_a_fazer = iter([(2, 4), (2, 4), (0, 2), (0, 2), (0, 2), (4, 5)])

    def retornar_cortes_na_ordem_especificada(*args, **kwargs):
        return next(cortes_a_fazer)

    with monkeypatch.context() as m:
        m.setattr(random, "choice", retornar_cortes_na_ordem_especificada)
        ambiente_de_teste.crossover(proj_1, proj_2)

    assert np.all(proj_1.gene[0:2, 4]) and np.all(~proj_1.gene[:, :4])
    assert np.all(proj_2.gene[0:2, 0:2]) and np.all(proj_2.gene[0:2, 5])




