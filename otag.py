import pickle
from pathlib import Path
from random import seed, getstate, setstate

import numpy as np
import pandas as pd
from matplotlib.pyplot import imshow, plot, colorbar, show, legend, clf, savefig

from situações_de_projeto.placa_em_balanço.placa_em_balanço import PopulaçãoDeProjetos

semente = 0
info_gerações = []


def rodar_teste(n=1):
    mudar_semente(0)
    pop = PopulaçãoDeProjetos()
    pop.avançar_gerações(n)


def mudar_semente(sem):
    global semente

    seed(sem)
    np.random.seed(sem)
    semente = sem


def salvar_estado(pop):
    raiz = Path.cwd()
    pasta_da_semente = "semente_{}".format(semente)
    pasta_da_geração = "geração_{}".format(pop.n_da_geração)
    caminho          = raiz / "dados" / pasta_da_semente / pasta_da_geração

    caminho.mkdir(parents=True, exist_ok=True)

    with open(caminho / "população.b", "wb") as backup:
        pickle.dump(pop, backup)

    with open(caminho / "estado_do_numpy.b", "wb") as backup:
        pickle.dump(np.random.get_state(), backup)

    with open(caminho / "estado_do_random.b", "wb") as backup:
        pickle.dump(getstate(), backup)


def carregar_estado(semente=0, geração=1):
    raiz = Path.cwd()
    pasta_da_semente = "semente_{}".format(semente)
    pasta_da_geração = "geração_{}".format(geração)
    caminho = raiz / "dados" / pasta_da_semente / pasta_da_geração

    try:
        with open(caminho / "população.b", "rb") as backup:
            pop = pickle.load(backup)

        with open(caminho / "estado_do_numpy.b", "rb") as backup:
            estado = pickle.load(backup)
            np.random.set_state(estado)

        with open(caminho / "estado_do_random.b", "rb") as backup:
            estado = pickle.load(backup)
            setstate(estado)

        print("> Estados dos geradores de números aleatórios redefinidos para quando o backup foi feito")

        return pop

    except FileNotFoundError:
        print("> Não há registros da geração {} começada com a semente {}".format(geração, semente))


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
    m_conv = np.vectorize(lambda x: 4*(x**2) - 4*x + 1)
    i_conv = sum(m_conv(conv).flat)/len(conv.flat)
    print("Índice de Convergência: {:.2f}%".format(100*i_conv))
    imshow(conv, cmap="hot")
    colorbar()
    show()


def mostrar_indivíduo(i, pop, tipo="malha", k=1):
    proj = pop.indivíduos[i]

    if tipo == "malha":
        proj.malha.plot(proj.u, k)

    elif tipo == "gene":
        imshow(~proj.gene, cmap="hot")
        colorbar()
        show()


def ciclo_de_(n, pop):
    pop.avançar_gerações(n)
    filtrar_informações(pop)
    salvar_estado(pop)


def execução_típica(n=100, pop=None, semente=0):
    retornar = False
    if pop is None:
        retornar = True
        mudar_semente(semente)
        pop = PopulaçãoDeProjetos()

    for k in range(pop.n_da_geração, pop.n_da_geração + n):
        pop.próxima_geração()

        if k % 10 == 0:
            filtrar_informações(pop)

        if k < 100 and k % 10 == 0:
            salvar_estado(pop)
        elif k % 100 == 0:
            salvar_estado(pop)

    if retornar:
        return pop


def salvar_resultado(pop):
    salvar_estado(pop)

    caminho = Path.cwd() / "resultados"

    try:
        tabela = pd.read_csv(caminho / "comparação_de_resultados.csv")
    except FileNotFoundError:
        tabela = pd.DataFrame(columns=["Semente", "Gerações", "Indivíduo_mais_apto",
                                       "Adaptação", "Índice_de_Convergência", "alfa_0", "e"])

    sem  = semente
    ger  = pop.n_da_geração
    prj  = pop.indivíduos[0]
    ima  = prj.nome
    adpt = prj.adaptação
    alfa = pop.alfa_0
    edes = 0.4

    conv   = sum([ind.gene for ind in pop.indivíduos]) / 100
    m_conv = np.vectorize(lambda x: 4 * (x ** 2) - 4 * x + 1)
    idc    = sum(m_conv(conv).flat) / len(conv.flat)

    linha = pd.DataFrame({"Semente":                [sem],
                          "Gerações":               [ger],
                          "Indivíduo_mais_apto":    [ima],
                          "Adaptação":             [adpt],
                          "Índice_de_Convergência": [idc],
                          "alfa_0":                [alfa],
                          "e":                     [edes]})

    tabela = tabela.append(linha)
    tabela.drop_duplicates(inplace=True)
    tabela.to_csv(caminho / "comparação_de_resultados.csv", index=False)

    imshow(~prj.gene, cmap="hot")
    savefig(caminho / ("gene_sem{}_{}.png".format(sem, prj.nome)), dpi=200)
    clf()

    prj.malha.plot(prj.u, show=False)
    savefig(caminho / ("malha_sem{}_{}.png".format(sem, prj.nome)), dpi=200)
    clf()

def execução_completa():
    for semente in range(10 + 1):
        pop = execução_típica(n=300, semente=semente)
        salvar_resultado(pop)

mudar_semente(semente)
print("> Semente mudada para o valor padrão (0)")