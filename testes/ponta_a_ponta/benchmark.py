from otag import conseguir_construtores

Ambiente, Problema, parâmetros = conseguir_construtores()
amb = Ambiente(Problema(parâmetros))


def teste_uma_geração_padrão():
    amb.próxima_geração()
    assert amb is not None
