from otag import carregar_estado
from visualizador.placa_em_balanço import *


amb  = carregar_estado(geração=1)
proj = amb.população[0]


def teste_mostrar_projeto(k=1):
    mostrar_projeto(proj, k=k)


if __name__ == "__main__":
    teste_mostrar_projeto(300)