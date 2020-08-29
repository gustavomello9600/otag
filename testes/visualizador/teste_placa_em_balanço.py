from otag import carregar_estado, mudar
from visualizador.placa_em_balanço import *


mudar("problema", "P_no_meio_da_extremidade_direita_e_cantos_das_bordas_fixos.py")

amb  = carregar_estado(semente=10, geração=300)
proj = amb.população[0]


def teste_mostrar_projeto(k=1):
    mostrar_projeto(proj, k=k)


def teste_plotar_com_cores(k=1, paleta="magma"):
    plotar_malha_com_cores(proj, k=k, paleta=paleta)


def teste_mostrar_ambiente():
    mostrar_ambiente(amb, semente=10)


if __name__ == "__main__":
    teste_mostrar_ambiente()
