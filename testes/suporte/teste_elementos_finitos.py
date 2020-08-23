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


if __name__ == "__main__":
    teste_Nó()