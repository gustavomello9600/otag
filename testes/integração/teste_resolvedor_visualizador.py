import numpy as np

from otag import carregar_estado
from visualizador.placa_em_balanço import mostrar_projeto

amb = carregar_estado(semente=0, geração=1)
proj = amb.população[0]


def teste_se_há_um_nó_por_etiqueta():
    etiquetas = {nó.etiqueta for nó in proj.malha.nós}

    assert len(etiquetas) == len(proj.malha.nós), f"{len(etiquetas)} != {len(proj.malha.nós)}"


def teste_se_me_conversa_com_método_índice_de():
    for e, elemento in enumerate(proj.malha.elementos):
        iul, iur, idr, idl = [proj.malha.índice_de[nó] for nó in elemento.nós]
        índices_globais = np.array([2*iul,
                                    2*iul + 1,
                                    2*iur,
                                    2*iur + 1,
                                    2*idr,
                                    2*idr + 1,
                                    2*idl,
                                    2*idl + 1], dtype="int16")

        assert np.all(índices_globais == proj.malha.me[:, e]), f"IGs: {índices_globais}\n.me: {proj.malha.me[:, e]}"

    print(f"IGs: {índices_globais}\nme: {proj.malha.me[:, e]}")


if __name__ == "__main__":
    teste_se_me_conversa_com_método_índice_de()