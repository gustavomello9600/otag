import pickle
from typing import Any, Tuple, FrozenSet
from pathlib import Path
from math import isclose
from dataclasses import dataclass, field

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class Nó:

    x: float = field(hash=False)
    y: float = field(hash=False)
    etiqueta: Any = field(default=None, hash=True)

    def __eq__(self, other):
        tol = 1e-10
        return (isclose(self.x, other.x, rel_tol=tol) and isclose(self.y, other.y, rel_tol=tol))


@dataclass
class Elemento:

    nós: Tuple[Nó]


@dataclass
class Malha:

    elementos: Tuple[Elemento]
    nós: Tuple[Nó]
    me: ArrayLike

    def __post_init__(self):
        self.ne = len(self.elementos)

    def índice_de(self, nó):
        return self.nós.index(nó)


class KeBase:

    def __init__(self):
        matriz, dicionário_de_variáveis = self.construir()

        # Atributos definidos acima
        self.matriz = matriz
        self.símbolo_de = dicionário_de_variáveis

    # Sobrescrever método
    def construir(self):
        pass

    def calcular(self, valor_de):
        parâmetros = valor_de.keys()
        correspondência = {self.símbolo_de[p]: valor_de[p] for p in parâmetros}
        return np.array(self.matriz.evalf(subs=correspondência), dtype=float)

    @classmethod
    def pronta(cls, cache="Ke_genérica.b"):
        caminho_para_o_arquivo = Path(__file__).parent / "cache" / cache

        if not caminho_para_o_arquivo.exists():
            caminho_para_o_arquivo.parent.mkdir(parents=True, exist_ok=True)
            K_base = cls()
            with caminho_para_o_arquivo.open("wb") as base_binária:
                pickle.dump(K_base, base_binária)
        else:
            with caminho_para_o_arquivo.open("rb") as base_binária:
                K_base = pickle.load(base_binária)

        return K_base
