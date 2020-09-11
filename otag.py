import os
import json
import pickle
import random
from pathlib import Path
from importlib import import_module
from typing import Optional, Sequence, Tuple, Dict, Union, Any, Type

import numpy as np
import pandas as pd

from visualizador.placa_em_balanço import calcular_convergência, mostrar_ambiente
from suporte.elementos_finitos.definição_de_problema import Problema
from suporte.algoritmo_genético import Ambiente


ClasseAmbiente = Type[Ambiente]
ClasseProblema = Type[Problema]
ConjuntoDeConstrutores = Tuple[ClasseAmbiente, ClasseProblema, Dict[str, Union[str, int, float]]]


raiz = Path(__file__).parent

semente = 0
info_gerações = []

situação_de_projeto = "placa_em_balanço"
parâmetros = "padrão.json"
ambiente = "Kane_e_Schoenauer.py"
problema = "P_no_meio_da_extremidade_direita.py"


def mudar(variável: str, valor: Any = "valor", interativo: bool = False) -> None:
    if interativo:
        if variável == "situação_de_projeto":
            valor = _listar_e_escolher("situações_de_projeto", situação_escolhida=False)
        else:
            valor = _listar_e_escolher(variável if variável.endswith("s") else variável + "s", situação_de_projeto)

    globals().update({variável: valor})


def interativo(padrão: bool = False,
               gerações: int = 20,
               execução_longa: bool = False,
               sementes: Optional[Sequence[int]] = None
               ) -> Ambiente:

    global situação_de_projeto, parâmetros, ambiente, problema

    if not padrão:
        situação_de_projeto = _listar_e_escolher("situações_de_projeto", situação_escolhida=False)
        parâmetros          = _listar_e_escolher("parâmetros", situação_de_projeto)
        problema            = _listar_e_escolher("problemas", situação_de_projeto)
        ambiente            = _listar_e_escolher("ambientes", situação_de_projeto)

    AmbienteDeProjeto, ProblemaDefinido, parâmetros_do_problema = conseguir_construtores()

    if execução_longa:
        amb = execução_completa(construtores=(AmbienteDeProjeto, ProblemaDefinido, parâmetros_do_problema))
    elif sementes is not None:
        for sem in sementes:
            amb = execução_típica(gerações,
                                  sem=sem,
                                  construtores=(AmbienteDeProjeto, ProblemaDefinido, parâmetros_do_problema))
    else:
        amb = execução_típica(gerações, construtores=(AmbienteDeProjeto, ProblemaDefinido, parâmetros_do_problema))

    return amb


def _listar_e_escolher(categoria: str,
                       situação_de_projeto: str = "None",
                       situação_escolhida: bool = True
                       ) -> str:

    caminho = ((raiz / "situações_de_projeto" / situação_de_projeto / categoria) if situação_escolhida
               else (raiz / categoria))

    alternativas = [alt for alt in os.listdir(caminho) if not alt.startswith("__")]

    opções = [f"  {i + 1}. {_formatação_casos_mistos(alt[:alt.index('.')])}"
              if "." in alt
              else f"  {i + 1}. {_formatação_casos_mistos(alt)}"
              for i, alt in enumerate(alternativas)]

    print(f"> Escolha dentre os(as) {_formatação_casos_mistos(categoria)}")
    for opção in opções:
        print(opção)
    optada = int(input(">>>"))

    return alternativas[optada - 1]


_exceções = ["em", "de", "do", "dos", "da", "das", "no", "nos", "na", "nas", "e", "o", "os", "a", "as"]
def _formatação_casos_mistos(s: str, reunir: str = " ") -> str:
    palavras = s.split("_")
    palavras = [p.capitalize() if p not in _exceções else p for p in palavras]
    return reunir.join(palavras)


def conseguir_construtores() -> ConjuntoDeConstrutores:
    # Alcança e processa os parâmetros do problema
    with open(raiz / "situações_de_projeto" / situação_de_projeto / "parâmetros" / parâmetros,
              encoding="utf-8") as arquivo:
        parâmetros_do_problema = json.load(arquivo)
        _processar(parâmetros_do_problema)

    # Alcança a definição do problema
    nome_da_classe_do_problema = "".join(p.capitalize() for p in situação_de_projeto.split("_"))
    módulo_do_problema = import_module(f"situações_de_projeto.{situação_de_projeto}.problemas.{problema[:-3]}")
    ProblemaDefinido = getattr(módulo_do_problema, nome_da_classe_do_problema)

    # Alcança o ambiente de projeto
    módulo_do_ambiente = import_module(f"situações_de_projeto.{situação_de_projeto}.ambientes.{ambiente[:-3]}")
    AmbienteDeProjeto = getattr(módulo_do_ambiente, "AmbienteDeProjeto")

    return AmbienteDeProjeto, ProblemaDefinido, parâmetros_do_problema


def _processar(parâmetros_do_problema: Dict[str, str]) -> None:
    for k, v in parâmetros_do_problema.items():
        if v[0] not in "0123456789":
            continue
        elif "." in v or "e" in v:
            parâmetros_do_problema[k] = float(v)
        else:
            parâmetros_do_problema[k] = int(v)


def execução_completa(amb: Optional[Ambiente] = None,
                      construtores: Optional[ConjuntoDeConstrutores] = None
                      ) -> Ambiente:
    for semente in range(10 + 1):
        amb = execução_típica(300, construtores=construtores, sem=semente)
    return amb


def execução_típica(n: int,
                    sem: int = 0,
                    amb: Optional[Ambiente] = None,
                    construtores: Optional[ConjuntoDeConstrutores] = None
                    ) -> Ambiente:
    amb = _iniciar_ambiente(sem, amb, construtores)

    print(f">> Executando: \n"
          f"       {ambiente[:-3]}({problema[:-3]}({parâmetros[:-5]})),\n"
          f"       semente={sem}\n")

    for k in range(amb.n_da_geração, amb.n_da_geração + n):
        amb.próxima_geração()

        if k % 10 == 0:
            _filtrar_informações(amb)
        if k % 100 == 0:
            salvar_estado(amb)

        if _índice_de_convergência_limite_atingido(amb, limite=0.95):
            print("------------------------------------------------------\n"
                  "Evolução parada por índice de convergência superar 95%\n\n")
            break

    if hasattr(amb, "finalizar"):
        amb.finalizar()

    salvar_resultado(amb)

    return amb


def _iniciar_ambiente(sem: int, amb: Ambiente, construtores: ConjuntoDeConstrutores) -> Ambiente:
    if not ((amb is None) ^ (construtores is None)):
        raise ValueError(f"Um, e apenas um, dentre o Ambiente ou o Conjunto de Construtores deve ser fornecido. "
                         f"Ambiente fornecido: {amb is not None}. Construtores fornecidos: {construtores is not None}")
    if amb is None:
        mudar_semente(sem)
    else:
        global semente
        semente = sem

    if construtores:
        Amb, Prob, Param = construtores
        amb = Amb(Prob(Param))

    return amb


def mudar_semente(sem: int) -> None:
    global semente

    random.seed(sem)
    np.random.seed(sem)
    semente = sem


def _filtrar_informações(amb: Ambiente) -> None:
    global info_gerações
    for gen in amb.gerações:
        adpts = [ind.adaptação for ind in gen]
        info_gerações.append((np.max(adpts), np.mean(adpts), np.min(adpts)))
    amb.gerações.clear()


def salvar_estado(amb: Ambiente) -> None:
    caminho = _localização_dos_dados(amb.n_da_geração, semente)
    caminho.mkdir(parents=True, exist_ok=True)

    with open(caminho / "população.b", "wb") as backup:
        pickle.dump(amb, backup)

    with open(caminho / "estado_do_numpy.b", "wb") as backup:
        pickle.dump(np.random.get_state(), backup)

    with open(caminho / "estado_do_random.b", "wb") as backup:
        pickle.dump(random.getstate(), backup)


def _localização_dos_dados(geração: int, semente: int) -> Path:
    pasta_da_situação = raiz / "situações_de_projeto" / situação_de_projeto
    pasta_da_semente = f"semente_{semente}"
    pasta_da_geração = f"geração_{geração}"
    pasta_do_contexto = f"{ambiente[:-3]}_{problema[:-3]}" \
                        f"{('_' + parâmetros[:-5]) if parâmetros != 'padrão.json' else ''}"

    caminho = pasta_da_situação / "dados" / pasta_do_contexto / pasta_da_semente / pasta_da_geração

    return caminho


def _índice_de_convergência_limite_atingido(amb: 'Ambiente', limite: float = 0.95) -> bool:
    índice_de_convergência: float
    if hasattr(amb, "índice_de_convergência"):
        índice_de_convergência = amb.índice_de_convergência
    else:
        _, índice_de_convergência = calcular_convergência(amb)

    return índice_de_convergência >= limite


def salvar_resultado(amb: Ambiente) -> None:
    salvar_estado(amb)

    pasta_do_contexto = f"{ambiente[:-3]}_{problema[:-3]}" \
                        f"{('_' + parâmetros[:-5]) if parâmetros != 'padrão.json' else ''}"

    caminho = raiz / "situações_de_projeto" / situação_de_projeto / "resultados" / pasta_do_contexto
    caminho.mkdir(parents=True, exist_ok=True)

    if (caminho / "comparação_de_resultados.csv").exists():
        tabela = pd.read_csv(caminho / "comparação_de_resultados.csv")
    else:
        tabela = pd.DataFrame(columns=["Semente", "Gerações", "Indivíduo_mais_apto",
                                       "Adaptação", "Índice_de_Convergência", "alfa_0", "e"])

    prj = amb.população[0]
    conv, idc = calcular_convergência(amb)

    linha = pd.DataFrame({"Semente": [semente],
                          "Gerações": [amb.n_da_geração],
                          "Indivíduo_mais_apto": [prj.nome],
                          "Adaptação": [prj.adaptação],
                          "Índice_de_Convergência": [idc],
                          "alfa_0": [amb.problema.alfa_0],
                          "e": [amb.problema.e]})

    tabela = tabela.append(linha)
    tabela.drop_duplicates(inplace=True)
    tabela.to_csv(caminho / "comparação_de_resultados.csv", index=False)

    mostrar_ambiente(amb, semente=semente, arquivo=(caminho / f"semente_{semente}.png"))


def carregar_estado(semente: int = 0, geração: int = 1) -> Optional[Ambiente]:
    caminho = _localização_dos_dados(geração, semente)

    try:
        with open(caminho / "população.b", "rb") as backup:
            amb = pickle.load(backup)

        with open(caminho / "estado_do_numpy.b", "rb") as backup:
            estado = pickle.load(backup)
            np.random.set_state(estado)

        with open(caminho / "estado_do_random.b", "rb") as backup:
            estado = pickle.load(backup)
            random.setstate(estado)

        print("> Estados dos geradores de números aleatórios redefinidos para quando o backup foi feito")

        return amb

    except FileNotFoundError:
        print(f"> Não há registros da geração {geração} começada com a semente {semente}")
