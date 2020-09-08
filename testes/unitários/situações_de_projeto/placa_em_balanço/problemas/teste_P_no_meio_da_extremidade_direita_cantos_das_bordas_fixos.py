import random
from unittest.mock import Mock

import pytest

from situações_de_projeto.placa_em_balanço.problemas.P_no_meio_da_extremidade_direita_e_cantos_das_bordas_fixos import *


@pytest.fixture
def placa_em_balanço(parâmetros_de_teste):
    return PlacaEmBalanço(parâmetros_de_teste)


@pytest.fixture
def projeto_teste(placa_em_balanço):
    proj = Mock()
    proj.nome = "ProjetoTeste"

    random.seed(0)
    np.random.seed(0)
    proj.gene = placa_em_balanço.geração_0(n_de_indivíduos=1, espessura_interna_mínima=2)[0]

    return proj


def teste_geração_0(placa_em_balanço):
    genes = placa_em_balanço.geração_0(n_de_indivíduos=4, espessura_interna_mínima=2)

    for gene in genes:
        assert gene[0, 0] and gene[-1, 0]
        assert gene[int(placa_em_balanço.n // 2), 2*placa_em_balanço.n - 1]


def teste_testar_adaptação(placa_em_balanço, capsys, projeto_teste):
    placa_em_balanço.testar_adaptação(projeto_teste)
    assert projeto_teste.adaptação_testada
    assert projeto_teste.adaptação == 0.9438414346389805

    projeto_teste.gene[-1, 0] = False
    placa_em_balanço.testar_adaptação(projeto_teste)
    saída_da_execução = capsys.readouterr().out
    assert f"> Projeto {projeto_teste.nome} desconectado dos cantos da borda" in saída_da_execução



