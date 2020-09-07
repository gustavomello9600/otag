import numpy as np
import pytest

from otag import carregar_estado


@pytest.fixture
def projeto():
    ambiente = carregar_estado(semente=0, geração=1)
    return ambiente.população[0]


def teste_se_há_um_nó_por_etiqueta(projeto):
    etiquetas = {nó.etiqueta for nó in projeto.malha.nós}

    assert len(etiquetas) == len(projeto.malha.nós)


def teste_se_me_conversa_com_método_índice_de(projeto):
    for e, elemento in enumerate(projeto.malha.elementos):
        iul, iur, idr, idl = [projeto.malha.índice_de[nó] for nó in elemento.nós]
        índices_globais = np.array([2*iul,
                                    2*iul + 1,
                                    2*iur,
                                    2*iur + 1,
                                    2*idr,
                                    2*idr + 1,
                                    2*idl,
                                    2*idl + 1], dtype="int16")

        assert np.all(índices_globais == projeto.malha.me[:, e])