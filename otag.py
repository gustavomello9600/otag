import os
import pickle
from pathlib import Path
from importlib import import_module
from random import seed, getstate, setstate

import numpy as np
import pandas as pd

from visualizador.placa_em_balanço import mostrar_projeto
from situações_de_projeto.placa_em_balanço.placa_em_balanço import AmbienteDeProjeto

os.chdir(Path(__file__).parent)

semente = 0
info_gerações = []
situação_de_projeto = "placa_em_balanço"


def mudar_situação_de_projeto(situação=None):
    if situação is None:
        situações = os.listdir("situações_de_projeto")
        opções = [f"  {i + 1}. {_casos_mistos(sit)}" for i, sit in enumerate(situações)]
        print("> Escolha uma situação de projeto")
        for i, opção in enumerate(opções):
            print(opção)
        opt = int(input("> Escolha uma opção:")) - 1
        situação = situações[opt]

    sdp = import_module("situações_de_projeto." + situação + "." + situação)

    global situação_de_projeto
    situação_de_projeto = situação

    global AmbienteDeProjeto
    AmbienteDeProjeto = sdp.AmbienteDeProjeto


_exceções = ["em", "de", "do", "da", "no", "na"]
def _casos_mistos(s):
    palavras = s.split("_")
    palavras = [p.capitalize() if p not in _exceções else p for p in palavras]
    return " ".join(palavras)


def mudar_semente(sem):
    global semente

    seed(sem)
    np.random.seed(sem)
    semente = sem


def execução_completa():
    for semente in range(10 + 1):
        amb = execução_típica(n=300, semente=semente)


def execução_típica(n=100, amb=None, semente=0):
    retornar = False
    if amb is None:
        retornar = True
        mudar_semente(semente)
        amb = AmbienteDeProjeto()

    for k in range(amb.n_da_geração, amb.n_da_geração + n):
        amb.próxima_geração()

        if k % 10 == 0:
            filtrar_informações(amb)

        if k % 100 == 0:
            salvar_estado(amb)

    salvar_resultado(amb)

    if retornar:
        return amb


def filtrar_informações(amb):
    global info_gerações
    for gen in amb.gerações:
        adpts = [ind.adaptação for ind in gen]
        info_gerações.append((np.max(adpts), np.mean(adpts), np.min(adpts)))
    amb.gerações.clear()


def salvar_estado(amb):
    raiz = Path.cwd() / "situações_de_projeto" / situação_de_projeto
    pasta_da_semente = "semente_{}".format(semente)
    pasta_da_geração = "geração_{}".format(amb.n_da_geração)
    caminho          = raiz / "dados" / pasta_da_semente / pasta_da_geração

    caminho.mkdir(parents=True, exist_ok=True)

    with open(caminho / "população.b", "wb") as backup:
        pickle.dump(amb, backup)

    with open(caminho / "estado_do_numpy.b", "wb") as backup:
        pickle.dump(np.random.get_state(), backup)

    with open(caminho / "estado_do_random.b", "wb") as backup:
        pickle.dump(getstate(), backup)


def salvar_resultado(amb):
    salvar_estado(amb)

    caminho = Path.cwd() / "situações_de_projeto" / situação_de_projeto / "resultados"

    try:
        tabela = pd.read_csv(caminho / "comparação_de_resultados.csv")
    except FileNotFoundError:
        caminho.mkdir(parents=True, exist_ok=True)
        tabela = pd.DataFrame(columns=["Semente", "Gerações", "Indivíduo_mais_apto",
                                       "Adaptação", "Índice_de_Convergência", "alfa_0", "e"])

    sem  = semente
    ger  = amb.n_da_geração
    prj  = amb.população[0]
    ima  = prj.nome
    adpt = prj.adaptação
    alfa = amb.problema.alfa_0
    edes = 0.4

    conv   = sum([ind.gene for ind in amb.população]) / 100
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

    mostrar_projeto(prj, arquivo=(caminho / f"semente_{sem}.png"))


def ciclo_de_(n, amb):
    amb.avançar_gerações(n)
    filtrar_informações(amb)
    salvar_estado(amb)


def carregar_estado(semente=0, geração=1):
    raiz = Path.cwd() / "situações_de_projeto" / situação_de_projeto
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


mudar_semente(semente)
print("> Semente mudada para o valor padrão (0)")