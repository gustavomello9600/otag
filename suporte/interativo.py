import pickle
import numpy as np
from random import seed
from matplotlib.pyplot import imshow, plot, colorbar, show, legend

semente = 0
info_gerações = []

def mudar_semente(sem):
    global semente

    seed(sem)
    np.random.seed(sem)
    semente = sem

def salvar_estado(pop):
   nome_do_arquivo = "PdP_semente{}_geração{}.b".format(semente, pop.geração)
       with open(nome_do_arquivo, "wb") as saída:
           pickle.dump(pop, saída)


def filtrar_informações(pop):
    global info_gerações
    for gen in pop.gerações:
        adpts = [ind.adaptação for ind in gen]
        info_gerações.append((np.max(adpts), np.mean(adpts), np.min(adpts)))
    pop.gerações.clear()


def mostrar_progresso(info):
   X = range(len(info))
   Ymax, Ymed, Ymin = zip(*info)
   plot(X, Ymax, "r--")
   plot(X, Ymed, "k-")
   plot(X, Ymin, "b--")
   legend(["Máxima", "Média", "Mínima"])
   show()

def mapa_de_convergência(pop):
    conv = sum([ind.gene for ind in pop.indivíduos])/100
    imshow(conv, cmap="hot")
    colorbar()
    show()

def ciclo_de_(n, pop):
    pop.avançar_gerações(n)
    filtrar_informações(pop)
    salvar_estado(pop)