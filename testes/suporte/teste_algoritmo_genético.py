from suporte.algoritmo_genético import Indivíduo


def teste_Indivíduo():
    ind1 = Indivíduo(12, "i1")
    ind2 = Indivíduo(10, "i2")

    ind1.adaptação = 12
    ind2.adaptação = 10

    assert ind1 > ind2
    assert ind1.nome == "i1"
    assert ind2.gene == 10
    assert hasattr(ind2, "id") and ind2.id == 10
    assert "i1: 12" == str(ind1)


if __name__ == "__main__":
    teste_Indivíduo()
