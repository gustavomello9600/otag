import os
import json
import pickle
from copy import deepcopy
from pathlib import Path
from importlib import import_module
from random import seed, getstate, setstate

import numpy as np
import pandas as pd

from visualizador.placa_em_balanço import mostrar_projeto

raiz = Path(__file__).parent

semente = 0
info_gerações = []

situação_de_projeto = "placa_em_balanço"
parâmetros = "padrão.json"
ambiente   = "Kane_&_Schoenauer.py"
problema   = "P_no_meio_da_extremidade_direita.py"

def interativo(padrão=False, execução_longa=False):
    if not padrão:
        global situação_de_projeto, parâmetros, ambiente, problema

        situações = os.listdir(raiz / "situações_de_projeto")
        opções = [f"  {i + 1}. {_casos_mistos(sit)}" for i, sit in enumerate(situações)]
        print("> Escolha uma situação de projeto")
        for opção in opções:
            print(opção)
        optada = int(input("> Escolha uma opção:")) - 1

        situação_de_projeto = situações[optada]

        possíveis_parâmetros = os.listdir(raiz / "situações_de_projeto" / situação_de_projeto / "parâmetros")
        opções = [f"  {i + 1}. {_casos_mistos(sit)}"[:-5] for i, sit in enumerate(possíveis_parâmetros)]
        print("> Escolha os parâmetros do problema")
        for opção in opções:
            print(opção)
        optada = int(input("> Escolha uma opção:")) - 1

        parâmetros = possíveis_parâmetros[optada]

        problemas = os.listdir(raiz / "situações_de_projeto" / situação_de_projeto / "problemas")
        opções = [f"  {i + 1}. {_casos_mistos(sit)}"[:-3] for i, sit in enumerate(problemas)]
        print("> Escolha a definição do problema")
        for opção in opções:
            print(opção)
        optada = int(input("> Escolha uma opção:")) - 1

        problema = problemas[optada]

        ambientes = os.listdir(raiz / "situações_de_projeto" / situação_de_projeto / "ambientes")
        opções = [f"  {i + 1}. {_casos_mistos(sit)}"[:-3] for i, sit in enumerate(ambientes)]
        print("> Escolha o ambiente de projeto")
        for opção in opções:
            print(opção)
        optada = int(input("> Escolha uma opção:")) - 1

        ambiente = ambientes[optada]

    with open(raiz / "situações_de_projeto" / situação_de_projeto / "parâmetros" / parâmetros) as arquivo:
        parâmetros_do_problema = json.load(arquivo)
    processar(parâmetros_do_problema)

    módulo_do_problema = import_module(f"situações_de_projeto.{situação_de_projeto}.problemas.{problema[:-3]}")
    ProblemaDefinido = getattr(módulo_do_problema, "".join(p.capitalize() for p in situação_de_projeto.split("_")))

    módulo_do_ambiente = import_module(f"situações_de_projeto.{situação_de_projeto}.ambientes.{ambiente[:-3]}")
    AmbienteDeProjeto = getattr(módulo_do_ambiente, "AmbienteDeProjeto")

    amb = AmbienteDeProjeto(ProblemaDefinido(parâmetros_do_problema))

    if execução_longa:
        execução_completa(amb)

    else:
        execução_típica(5, amb)


def processar(parâmetros_do_problema):
    for k, v in parâmetros_do_problema.items():
        if v[0] not in "0123456789":
            continue
        elif "." in v or "e" in v:
            parâmetros_do_problema[k] = float(v)
        else:
            parâmetros_do_problema[k] = int(v)



_exceções = ["em", "de", "do", "da", "no", "na"]
def _casos_mistos(s, reunir=" "):
    palavras = s.split("_")
    palavras = [p.capitalize() if p not in _exceções else p for p in palavras]
    return reunir.join(palavras)


def mudar_semente(sem):
    global semente

    seed(sem)
    np.random.seed(sem)
    semente = sem


def execução_completa(amb):
    for semente in range(10 + 1):
        execução_típica(300, deepcopy(amb), semente=semente)


def execução_típica(n, amb, semente=0):
    for k in range(amb.n_da_geração, amb.n_da_geração + n):
        amb.próxima_geração()

        if k % 10 == 0:
            filtrar_informações(amb)

        if k % 100 == 0:
            salvar_estado(amb)

    salvar_resultado(amb)

    return amb


def filtrar_informações(amb):
    global info_gerações
    for gen in amb.gerações:
        adpts = [ind.adaptação for ind in gen]
        info_gerações.append((np.max(adpts), np.mean(adpts), np.min(adpts)))
    amb.gerações.clear()


def salvar_estado(amb):
    pasta_da_situação = raiz / "situações_de_projeto" / situação_de_projeto
    pasta_da_semente = "semente_{}".format(semente)
    pasta_da_geração = "geração_{}".format(amb.n_da_geração)
    caminho          = pasta_da_situação / "dados" / f"{ambiente}__{problema}" / pasta_da_semente / pasta_da_geração

    caminho.mkdir(parents=True, exist_ok=True)

    with open(caminho / "população.b", "wb") as backup:
        pickle.dump(amb, backup)

    with open(caminho / "estado_do_numpy.b", "wb") as backup:
        pickle.dump(np.random.get_state(), backup)

    with open(caminho / "estado_do_random.b", "wb") as backup:
        pickle.dump(getstate(), backup)


def salvar_resultado(amb):
    salvar_estado(amb)

    caminho = raiz / "situações_de_projeto" / situação_de_projeto / "resultados" / f"{ambiente}__{problema}"

    caminho.mkdir(parents=True, exist_ok=True)

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
    pasta_da_situação = raiz / "situações_de_projeto" / situação_de_projeto
    pasta_da_semente = "semente_{}".format(semente)
    pasta_da_geração = "geração_{}".format(geração)
    caminho = pasta_da_situação / "dados" / f"{ambiente}__{problema}" / pasta_da_semente / pasta_da_geração

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