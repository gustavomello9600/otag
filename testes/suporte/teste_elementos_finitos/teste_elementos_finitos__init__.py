from suporte.elementos_finitos import *


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


if __name__ == "__main__":
    teste_hash_do_Nó()
