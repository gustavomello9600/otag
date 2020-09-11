from otag import mudar, conseguir_construtores


mudar("problema", "P_no_meio_da_extremidade_direita_e_cantos_das_bordas_fixos.py")
mudar("parâmetros", "pouca_espessura.json")

Ambiente, Problema, parâmetros = conseguir_construtores()
amb = Ambiente(Problema(parâmetros))


def teste_uma_geração_padrão():
    amb.próxima_geração()
