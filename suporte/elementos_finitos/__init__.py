"""Módulo base para implementação de algoritmos de análise por elementos finitos.

Define variáveis de tipo e classes usadas por todas as implementações de situações de projeto.

CLASSES
-------
Malha    -- Malha de elementos finitos.
Elemento -- Elemento finito.
Nó       -- Nó de um elemento finito.

KeBase   -- Classe abstrata base de matrizes de rigidez locais específicas para cada tipo de elemento.
"""

import pickle
from math import isclose
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Tuple, List, Dict, Union, MutableSequence, ClassVar

import sympy
import numpy as np


Real = Union[int, float]
Vetor = MutableSequence[Real]
Matriz = MutableSequence[Vetor]

MatrizSimbólica = sympy.Matrix
SímboloDeVariável = sympy.Symbol


@dataclass
class Malha:
    """Malha de elementos finitos.

    Essa classe define os dados que as Malhas carregam em seus atributos.

    ATRIBUTOS
    ---------
    elementos: List['Elemento'] -- Lista que carrega cada elemento da malha na posição correspondente a seu índice
    nós      : List['Nó']       -- Lista que carrega cada nó da malha na posição correspondente a seu índice global
    me       : Matriz           -- Matriz de correspondência entre índices locais e globais de cada nó em cada elemento
    ne       : int              -- Número de elementos da malha
    índice_de: Dict['Nó', int]  -- Correspondência entre cada nó e seu indice global
    """

    elementos: List['Elemento']
    nós: List['Nó']
    me: Matriz

    def __post_init__(self):
        self.ne: int = len(self.elementos)
        self.índice_de: Dict['Nó', int] = {nó: i for i, nó in enumerate(self.nós)}


@dataclass
class Elemento:
    """Elemento finito.

    Classe que carrega nós devidamente ordenados. Sua estrutura deve ser implementada por suas subclasses."""

    nós: Tuple['Nó', ...]


@dataclass(frozen=True)
class Nó:
    """Nó de um elemento finito.

    Classe que carrega as coordenadas de um nó, uma etiqueta identificadora usada para tornar a classe hasheável (e por
    consequência passível de ser usada como chave de um dicionário) e a tolerância adotada quando é necessário comparar
    duas instâncias pelas suas coordenadas.
    """

    tolerância_na_comparação: ClassVar[float] = 1e-10

    x: float = field(hash=False)
    y: float = field(hash=False)
    etiqueta: Any = field(default=None, hash=True)

    def __eq__(self, other):
        if self.etiqueta == other.etiqueta:
            return True
        else:
            return (isclose(self.x, other.x, rel_tol=self.tolerância_na_comparação)
                    and isclose(self.y, other.y, rel_tol=self.tolerância_na_comparação))


class KeBase(ABC):
    """Classe abstrata base de matrizes de rigidez locais específicas para cada tipo de elemento finito.

    Suas subclasses devem definir como são construídas inicialmente ao sobrescrever o método construir. Uma vez prontas,
    são salvas em cache num arquivo binário. Cada subclasse tem uma única instância criada no momento que seu módulo é
    importado e usada pelas definições de problema para calcular a matriz de rigidez local usando os parâmetros por elas
    fornecidas.

    ATRIBUTOS
    ---------
    matriz    : MatrizSimbólica              -- Matriz de Rigidez Local parametrizada pelas coordenadas naturais do ele-
                                                mento e pelas suas propriedades materiais.
    símbolo_de: Dict[str, SímboloDeVariável] -- Correspondência entre a versão em string de cada parâmetro e sua versão
                                                em símbolo da biblioteca sympy.

    MÉTODOS (da classe)
    -------
    pronta(cache: str = "Ke_genérica.b") -> 'KeBase'
        Recupera de um cache binário a matriz simbólica já construída e pronta para uso. Caso o cache não exista, uma
        nova instância é criada.

    MÉTODOS ABSTRATOS (das instâncias)
    -----------------
    construir() -> Tuple[MatrizSimbólica, Dict[str, SímboloDeVariável]]
        Construirá, a partir de manipulações algébricas, a matriz de rigidez local do elemento finito correspondente,
        retornando-la junto a um dicionário de parâmetros.

    MÉTODOS CONCRETOS (das instâncias)
    -----------------
    calcular(valor_de: Dict[str, Real]) -> Matriz
        Recebe os valores reais de cada parâmetro e retorna a matriz de rigidez local numérica.
    """

    @classmethod
    def pronta(cls, cache: str = "Ke_genérica.b") -> 'KeBase':
        """Recupera de um cache binário a matriz simbólica já construída e pronta para uso. Caso o cache não exista, uma
        nova instância é criada."""
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

    def __init__(self):
        matriz, dicionário_de_variáveis = self.construir()

        # Atributos definidos acima
        self.matriz: MatrizSimbólica = matriz
        self.símbolo_de: Dict[str, SímboloDeVariável] = dicionário_de_variáveis

    @abstractmethod
    def construir(self) -> Tuple[MatrizSimbólica, Dict[str, SímboloDeVariável]]:
        """Construirá, a partir de manipulações algébricas, a matriz de rigidez local do elemento finito correspondente,
        retornando-la junto a um dicionário de parâmetros."""

    def calcular(self, valor_de: Dict[str, Real]) -> Matriz:
        """Recebe os valores reais de cada parâmetro e retorna a matriz de rigidez local numérica."""
        parâmetros = valor_de.keys()
        correspondência = {self.símbolo_de[p]: valor_de[p] for p in parâmetros}
        return np.array(self.matriz.evalf(subs=correspondência), dtype=float)
