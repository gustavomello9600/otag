from unittest.mock import Mock

import pytest

from situações_de_projeto.placa_em_balanço.problemas.P_no_meio_da_extremidade_direita import *


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


def teste_inicialização_de_instância_do_problema(parâmetros_de_teste):
    placa_em_balanço = PlacaEmBalanço(parâmetros_de_teste)

    assert placa_em_balanço.n == 20
    assert placa_em_balanço._método_padrão == "OptV2"
    assert placa_em_balanço.Dlim == 0.005
    assert placa_em_balanço.alfa_0 == 10
    assert placa_em_balanço.e == 0.1
    assert placa_em_balanço.lado_dos_elementos == 1 / 20
    assert placa_em_balanço.alfa == 10


def teste_ordem_de_refinamento_da_malha_inválida(parâmetros_de_teste):
    parâmetros_de_teste["ORDEM_DE_REFINAMENTO_DA_MALHA"] = 19
    with pytest.raises(ValueError):
        placa_em_balanço = PlacaEmBalanço(parâmetros_de_teste)

    parâmetros_de_teste["ORDEM_DE_REFINAMENTO_DA_MALHA"] = 6
    with pytest.raises(ValueError):
        placa_em_balanço = PlacaEmBalanço(parâmetros_de_teste)


def teste_geração_0(placa_em_balanço):
    genes = placa_em_balanço.geração_0(n_de_indivíduos=4, espessura_interna_mínima=2)

    for gene in genes:
        assert gene[:, 0].any()
        assert gene[int(placa_em_balanço.n // 2), 2*placa_em_balanço.n - 1]


def teste_testar_adaptação(placa_em_balanço, capsys, projeto_teste):
    placa_em_balanço.testar_adaptação(projeto_teste)
    assert projeto_teste.adaptação_testada
    assert projeto_teste.adaptação == 0.7352941176470587

    placa_em_balanço.testar_adaptação(projeto_teste)
    saída_da_execução = capsys.readouterr().out
    assert f"> Adaptação de {projeto_teste.nome} já era conhecida pelo seu fenótipo" in saída_da_execução

    projeto_teste.gene[:, 0] = False
    placa_em_balanço.testar_adaptação(projeto_teste)
    saída_da_execução = capsys.readouterr().out
    assert f"> Projeto {projeto_teste.nome} desconectado da borda" in saída_da_execução


def teste_penalização(placa_em_balanço, projeto_teste, capsys):
    placa_em_balanço.Dlim = 0.0000001

    placa_em_balanço.testar_adaptação(projeto_teste)
    saída_da_execução = capsys.readouterr().out
    assert f"> Projeto {projeto_teste.nome} penalizado: Dmax - Dlim = " in saída_da_execução


@pytest.mark.parametrize("método",
    [pytest.param("expansão", marks=pytest.mark.skip(reason="Pouco utilizado e computacionalmente custoso")),
     pytest.param("compacto", marks=pytest.mark.skip(reason="Pouco utilizado e computacionalmente custoso")),
     pytest.param("OptV1", marks=pytest.mark.skip(reason="Pouco utilizado e computacionalmente custoso")),
     pytest.param("OptV2")])
def teste_montadores(placa_em_balanço, projeto_teste, método):
    placa_em_balanço._método_padrão = método

    placa_em_balanço.testar_adaptação(projeto_teste)
    assert projeto_teste.adaptação_testada
    assert projeto_teste.adaptação == 0.7352941176470587


