import random
from unittest.mock import Mock

import numpy as np
import matplotlib.pyplot as plt

from situações_de_projeto.placa_em_balanço.problemas.P_no_meio_da_extremidade_direita_e_cantos_das_bordas_fixos import (
    PlacaEmBalanço)
from visualizador.placa_em_balanço import plotar_gene


def teste_geração_0():
    geração_0 = PlacaEmBalanço.geração_0

    genes = geração_0(PlacaEmBalanço)

    assert len(genes) == 125
    assert isinstance(genes[0], np.ndarray)

    if __name__ == "__main__":
        np.random.seed(0)
        random.seed(0)
        for gene in genes:
            proj = Mock()
            proj.gene = gene
            figura, gráfico = plt.subplots()
            plotar_gene(proj, gráfico)
            plt.show()


if __name__ == '__main__':
    teste_geração_0()