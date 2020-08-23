from otag import carregar_estado
from visualizador.placa_em_balanço import *

amb  = carregar_estado(geração=11)
proj = amb.população[0]

def teste_mostrar_projeto():
    mostrar_projeto(proj)

if __name__ == "__main__":
    teste_mostrar_projeto()