from situações_de_projeto.placa_em_balanço.placa_em_balanço import AmbienteDeProjeto
from otag import mudar_semente

mudar_semente(0)
amb = AmbienteDeProjeto()
amb.avançar_gerações(1)