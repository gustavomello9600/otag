from suporte.elementos_finitos import *
from sympy import *


def teste_Nó():

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

    assert hash(nó) == hash(("identificador", )), f"Hash do Nó:       {hash(nó)}\n"    \
                                              f"Hash da etiqueta: {hash((etiqueta,))}"

    d = dict()
    d[nó] = 3

    assert d[Nó(0, 0, etiqueta)] == 3


def teste_KeBase():
    class KeBaseImplementada(KeBase):
        def construir(self):
            a, b = symbols("a b")

            matriz = Matrix([[a, 0], [0, b]])
            dicionário_de_variáveis = {"a": a, "b": b}

            return matriz, dicionário_de_variáveis

    Ke_base = KeBaseImplementada()

    assert isinstance(Ke_base.matriz, Matrix)
    assert isinstance(Ke_base.símbolo_de, dict)

    for k, v in Ke_base.símbolo_de.items():
        assert isinstance(k, str) and isinstance(v, Symbol), f"type({k}) = {type(k)}, type({v}) = {type(v)}"

    assert np.all(Ke_base.calcular({"a": 1, "b": 2}) == np.array([[1.0, 0.0], [0.0, 2.0]]))


if __name__ == '__main__':
    teste_Nó()
    teste_hash_do_Nó()
    teste_KeBase()
