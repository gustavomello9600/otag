from otag import carregar_estado
from visualizador.placa_em_balanço import *


amb  = carregar_estado(geração=20)
proj = amb.população[0]


def teste_mostrar_projeto(k=1):
    mostrar_projeto(proj, k=k)


def teste_plotar_com_cores(k=1, paleta="magma"):
    plotar_com_cores(proj, k=k, paleta=paleta)


if __name__ == "__main__":
    teste_plotar_com_cores(0, paleta="plasma")
