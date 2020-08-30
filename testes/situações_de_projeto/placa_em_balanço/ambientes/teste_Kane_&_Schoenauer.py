from unittest.mock import Mock

import numpy as np
import matplotlib.pyplot as plt

from otag import carregar_estado


amb = carregar_estado(semente=0, geração=1)

self = Mock()
self.probabilidade_de_mutar = 0.01/100


def teste_mutação_baseada_na_topologia(self, população):
    pm = 100 * self.probabilidade_de_mutar

    for ind in população:
        gene = ind.gene; ver(gene)
        bordas = ((gene ^ np.roll(gene, 1))
                  | (gene ^ np.roll(gene, -1))
                  | (gene ^ np.roll(gene, 1, axis=0))
                  | (gene ^ np.roll(gene, -1, axis=0))) ; ver(bordas)

        aumentar_bordas = True if 0.5 > np.random.random() else False
        if aumentar_bordas:
            bordas_sujeitas_a_mutação = gene & bordas
        else:
            bordas_sujeitas_a_mutação = ~gene & bordas
        ver(bordas_sujeitas_a_mutação)

        bits_virados = (bordas_sujeitas_a_mutação
                        & np.random.choice((True, False), bordas_sujeitas_a_mutação.shape, p=(pm, 1 - pm)))
        ver(bits_virados)

        gene[bits_virados] = ~gene[bits_virados]
        ver(gene)

        if np.any(bits_virados):
            ind.adaptação = 0
            ind.adaptação_testada = False
            print(f"{ind.nome} resetado")


def ver(matriz):
    plt.imshow(~matriz, cmap="hot")
    plt.show()
    esperar = input()


if __name__ == '__main__':
    teste_mutação_baseada_na_topologia(self, amb.população)