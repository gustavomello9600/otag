from otag import buscar_construtores

Ambiente, Problema, parâmetros = buscar_construtores()
amb = Ambiente(Problema(parâmetros))


def teste_uma_geração_padrão():
    amb.próxima_geração()
