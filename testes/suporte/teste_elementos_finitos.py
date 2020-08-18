from suporte.elementos_finitos import *


def teste_monitorar():

    res = Problema(dict())

    res.resolver_para(dict(), list(), monitorar=True)

    assert isinstance(res, Problema)


if __name__ == "__main__":
    teste_monitorar()
