import pytest

from suporte.algoritmo_genético import *


@pytest.fixture
def indivíduos_teste():
    ind1 = Indivíduo("1001", "G0_1", adaptação=9, adaptação_testada=True)
    ind2 = Indivíduo("1010", "G0_2", adaptação=10, adaptação_testada=True)
    ind3 = Indivíduo("1011", "G0_3", adaptação=11, adaptação_testada=False)
    ind4 = Indivíduo("1100", "G0_4", adaptação=12, adaptação_testada=False)
    return [ind1, ind2, ind3, ind4]


@pytest.fixture
def ambiente_teste(indivíduos_teste):
    class AmbienteTeste(Ambiente):

        def geração_0(self) -> List[Indivíduo]:
            return indivíduos_teste

        def testar_adaptação(self, indivíduo: Indivíduo) -> None:
            indivíduo.adaptação = sum([int(b)*(2**i) for i, b in enumerate(reversed(indivíduo.gene))])
            indivíduo.adaptação_testada = True

        def crossover(self, pai1: Indivíduo, pai2: Indivíduo, i: int) -> Indivíduo:
            return Indivíduo(pai1.gene[:2] + pai2.gene[2:], f"G{self.n_da_geração}_{i}")

        def mutação(self, geração: List[Indivíduo]) -> None:
            for i, ind in enumerate(geração):
                bit_virado = "0" if ind.gene[i] == "1" else "1"
                ind.gene = ind.gene[:i] + bit_virado + ind.gene[i+1:]

    return AmbienteTeste()


def teste_comparação_de_indivíduos(indivíduos_teste):
    ind1, ind2, ind3, ind4 = indivíduos_teste

    assert ind4 > ind3 > ind2 > ind1


def teste_avançar_geração(ambiente_teste, indivíduos_teste):
    resultante = [Indivíduo(gene='1100', nome='G0_4', adaptação=12, adaptação_testada=True),
                  Indivíduo(gene='1011', nome='G0_3', adaptação=11, adaptação_testada=True),
                  Indivíduo(gene='0000', nome='G0_1', adaptação=8, adaptação_testada=False),
                  Indivíduo(gene='1100', nome='G0_2', adaptação=8, adaptação_testada=False)]

    np.random.seed(0)
    ambiente_teste.avançar_gerações(1)

    assert ambiente_teste.n_da_geração == 1
    assert resultante == ambiente_teste.população
    assert [indivíduos_teste, resultante] == ambiente_teste.gerações
    
    
def teste_seleção_natural(ambiente_teste, indivíduos_teste):
    indivíduos_esperados = list(reversed(indivíduos_teste))[:2]
    indivíduos_retornados = ambiente_teste.seleção_natural()

    assert all([indivíduos_esperados[i].gene == indivíduos_retornados[i].gene for i in range(2)])
    assert all([indivíduos_esperados[i].nome == indivíduos_retornados[i].nome for i in range(2)])
    assert all([indivíduos_esperados[i].adaptação == indivíduos_retornados[i].adaptação for i in range(2)])

    assert all([indivíduos_retornados[i].adaptação_testada for i in range(2)])


def teste_reprodução(ambiente_teste, indivíduos_teste):
    np.random.seed(1)
    indivíduos_esperados = [Indivíduo(gene='1010', nome='G0_1', adaptação=10, adaptação_testada=True),
                            Indivíduo(gene='1010', nome='G0_2', adaptação=10, adaptação_testada=True)]
    indivíduos_retornados = ambiente_teste.reprodução(indivíduos_teste[:2])

    assert indivíduos_esperados == indivíduos_retornados

    assert all([indivíduos_esperados[i].gene == indivíduos_retornados[i].gene for i in range(2)])
    assert all([indivíduos_esperados[i].nome == indivíduos_retornados[i].nome for i in range(2)])
    assert all([indivíduos_esperados[i].adaptação == indivíduos_retornados[i].adaptação for i in range(2)])

    assert all([indivíduos_retornados[i].adaptação_testada for i in range(2)])


def teste_representação_do_ambiente(ambiente_teste):
    assert repr(ambiente_teste).startswith("Geração 0 de População de 4 indivíduos:")
    assert str(ambiente_teste).startswith("População de 4 indivíduos em sua geração 0:\n")
