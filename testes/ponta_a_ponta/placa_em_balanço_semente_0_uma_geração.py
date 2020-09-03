from otag import *


mudar("problema", "P_no_meio_da_extremidade_direita_e_cantos_das_bordas_fixos.py")
mudar("parâmetros", "pouca_espessura.json")


def limpar_execução():
    mudar_semente(0)


def teste_ponta_a_ponta(paralelização=False, mostrar=True):
    Ambiente, Problema, parâmetros_do_problema = conseguir_construtores()
    amb = Ambiente(Problema(parâmetros_do_problema),
                   n_de_indivíduos=40,
                   paralelização=paralelização)    
    amb.avançar_gerações(3)
    
    if hasattr(amb, "finalizar"):
        amb.finalizar()

    if mostrar:
        mostrar_ambiente(amb)

    Ambiente.genes_testados.clear()


if __name__ == '__main__':
    teste_ponta_a_ponta(paralelização=True)
