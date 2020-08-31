from otag import *


mudar("problema", "P_no_meio_da_extremidade_direita_e_cantos_das_bordas_fixos.py")
mudar("parâmetros", "pouca_espessura.json")


def teste_ponta_a_ponta():
    interativo(padrão=True, gerações=30)


if __name__ == '__main__':
    teste_ponta_a_ponta()