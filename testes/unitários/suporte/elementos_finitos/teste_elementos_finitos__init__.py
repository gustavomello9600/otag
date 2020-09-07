import pytest
from sympy import symbols, Matrix, Symbol

from suporte.elementos_finitos import *


@pytest.fixture
def Ke_base():
    class KeBaseImplementada(KeBase):
        def construir(self):
            a, b = symbols("a b")

            matriz = Matrix([[a, 0], [0, b]])
            dicionário_de_variáveis = {"a": a, "b": b}

            return matriz, dicionário_de_variáveis

    return KeBaseImplementada()


def teste_Malha_post_init():
    nós = [Nó(0, 0, etiqueta=(0, 0)), Nó(1, 1, etiqueta=(1, 1)),
           Nó(2, 0, etiqueta=(2, 0)), Nó(1, -1, etiqueta=(1, -1))]
    elementos = [Elemento((nós[0], nós[1], nós[3])),
                 Elemento((nós[1], nós[2], nós[3]))]
    me = [[0, 2],
          [1, 3],
          [2, 4],
          [3, 5],
          [6, 6],
          [7, 7]]

    malha = Malha(elementos, nós, me)

    assert malha.ne == 2
    assert malha.índice_de[Nó(1, 1, etiqueta=(1, 1))] == malha.me[2][0]/2 == malha.me[0][1]/2


def teste_igualdade_de_Nós():

    nó_1 = Nó(2, 1)
    nó_2 = Nó(2, 0.5)

    assert nó_1 != nó_2

    nó_3 = Nó(2, 1)
    nó_4 = Nó(2, 0.9999999)

    assert nó_3 != nó_4

    nó_5 = Nó(2, 1)
    nó_6 = Nó(2, 0.99999999999)

    assert nó_5 == nó_6


def teste_hash_do_Nó():
    etiqueta = "identificador"
    nó = Nó(0, 0, etiqueta=etiqueta)

    assert hash(nó) == hash(("identificador", ))

    d = dict()
    d[nó] = 3

    assert d[Nó(0, 0, etiqueta)] == 3


def teste_tipos_dos_atributos_de_KeBase(Ke_base):
    assert isinstance(Ke_base.matriz, Matrix)
    assert isinstance(Ke_base.símbolo_de, dict)

    for k, v in Ke_base.símbolo_de.items():
        assert isinstance(k, str) and isinstance(v, Symbol)


def teste_Ke_base_calcular(Ke_base):
    assert np.all(Ke_base.calcular({"a": 1, "b": 2}) == np.array([[1.0, 0.0], [0.0, 2.0]]))


def teste_Ke_base_pronta(Ke_base, monkeypatch):
    KeBaseImplementada = type(Ke_base)
    def retornar_verdadeiro(*args, **kwargs):
        return True

    def retornar_falso(*args, **kwargs):
        return False

    def retornar_maquete_de_arquivo(*args, **kwargs):
        class MaqueteDeContexto():
            def __enter__(self):
                return "maquete de arquivo"
            def __exit__(self, *args, **kwargs):
                pass

        return MaqueteDeContexto()

    def retornar_Ke_base(*args, **kwargs):
        return Ke_base

    def fazer_nada(*args, **kwargs):
        pass

    with monkeypatch.context() as patch:
        patch.setattr(Path, "exists", retornar_verdadeiro)
        patch.setattr(Path, "open", retornar_maquete_de_arquivo)
        patch.setattr(pickle, "load", retornar_Ke_base)

        assert Ke_base.matriz == KeBaseImplementada.pronta(cache="qualquer").matriz

    with monkeypatch.context() as patch:
        patch.setattr(Path, "exists", retornar_falso)
        patch.setattr(Path, "mkdir", fazer_nada)
        patch.setattr(Path, "open", retornar_maquete_de_arquivo)
        patch.setattr(pickle, "dump", fazer_nada)

        assert Ke_base.matriz == KeBaseImplementada.pronta(cache="qualquer").matriz
