from timeit import default_timer
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Tuple, Container, MutableSequence

import numpy as np
from numpy.linalg import solve

from suporte.elementos_finitos import Malha, Vetor, Matriz


Máscara = Union[MutableSequence[bool], slice, np.ndarray]


class Problema(ABC):
    """Framework de base para a definição de problemas.

    Define o método padrão de resolução de uma malha de elementos finitos sujeita a determinadas condições de contorno
    em u e f e implementa algumas funcionalidades úteis a qualquer classe de problema em particular.

    CLASSE INTERNA
    --------------
    Monitorador -- Decorador que mensura o tempo de execução do resolvedor e a duração de cada etapa.

    ATRIBUTOS
    ---------
    parâmetros_do_problema  : Dict[str, Union[str, int, float]] -- carrega as particularidades da instância
    _método_padrão          : Optional[str]                     -- determina o método padrão de montagem da matriz de
                                                                   rigidez geral
    _monitoramento_ativo    : bool                              -- usado para determinar a atividade do Monitorador
    _início_do_monitoramento: Optional[float]                   -- usado para calcular tempos de execução
    _última_medição         : Optional[float]                   -- usado para calcular tempos de execução

    MÉTODOS CONCRETOS
    -----------------
    resolver_para(parâmetros_dos_elementos: Dict[str, float],
                  malha: Malha, método: str = None, monitorar: bool = False) -> Tuple[Vetor, Vetor, Malha]
        Resolve a malha fornecida de acordo com os parâmetros dos seus elementos.

    MÉTODOS ABSTRATOS
    -----------------
    determinar_graus_de_liberdade(self, malha: Malha) -> int
        Para a malha fornecida, retornará o número total de graus de liberdade do conjunto de seus nós.
    calcular_matrizes_de_rigidez_local(**parâmetros_dos_elementos) -> Union[Matriz, Container[Matriz]]:
        Retornará, a partir dos parâmetros fornecidos, as matrizes de rigidez local de cada elemento.
    montar_matriz_de_rigidez_geral(malha: Malha,
                                   Ks_locais: Union[Matriz, Container[Matriz]],
                                   graus_de_liberdade: int,
                                   método: str
                                   ) -> Matriz:
        Montará a matriz de rigidez geral da malha usando o método especificado.
    incorporar_condições_de_contorno(malha: Malha, graus_de_liberdade: int, **parâmetros_do_problema
                                     ) -> Tuple[Vetor, Vetor, Máscara, Máscara]:
        Atualizará os valores dos vetores f e u para incorporar as condições de contorno.

    MÉTODOS AUXILIARES
    ------------------
    _configurar_monitoramento(monitorar: bool) -> None
        Se monitorar é True, liga o Monitorador e inicia seu timer.
    _desligar_monitoramento() -> None
        Desliga o Monitorador e reinicia o timer.
    _onde_f_é_conhecido_fatiar(self, K: Matriz, ifc: Máscara) -> Matriz
    _resolver_sistema_linear(self, Kfc: Matriz, f: Vetor, ifc: Máscara) -> Vetor
    _atualizar_graus_de_liberdade(u: Vetor, ifc: Máscara, ufc: Vetor) -> None
    _onde_u_é_conhecido_fatiar(K: Matriz, iuc: Máscara) -> Matriz
    _atualizar_valores_de(f: Vetor, iuc: Máscara, Kuc: Matriz, u: Vetor) -> None
    """

    def __init__(self,
                 parâmetros_do_problema: Dict[str, Union[str, int, float]],
                 método_padrão: Optional[str] = None):
        
        self.parâmetros_do_problema = parâmetros_do_problema
        self._método_padrão = método_padrão

        self._monitoramento_ativo = False
        self._início_do_monitoramento = None
        self._última_medição = None

    class Monitorador:
        """Decorador que mensura o tempo de execução do resolvedor e a duração de cada etapa"""

        def __init__(self, mensagem="Não definido"):
            self._mensagem = mensagem

        def __call__(self, método):
            def monitorador(problema, *args, **kwargs):
                if problema._monitoramento_ativo:
                    antes = default_timer()

                    retorno = método(problema, *args, **kwargs)

                    agora = default_timer()
                    no_método = agora - antes
                    desde_o_início = agora - problema._início_do_monitoramento

                    print(f"> {desde_o_início: >10.5f}. {self._mensagem} ({no_método:.5f})")

                    return retorno
                else:
                    return método(problema, *args, **kwargs)

            return monitorador

    def resolver_para(self,
                      parâmetros_dos_elementos: Dict[str, float],
                      malha: Malha,
                      método: str = None,
                      monitorar: bool = False
                      ) -> Tuple[Vetor, Vetor, Malha]:
        """Resolve a malha fornecida de acordo com os parâmetros dos seus elementos."""

        self._configurar_monitoramento(monitorar)

        graus_de_liberdade = self.determinar_graus_de_liberdade(malha)

        Ks_locais = self.calcular_matrizes_de_rigidez_local(**parâmetros_dos_elementos)
        K = self.montar_matriz_de_rigidez_geral(malha, Ks_locais, graus_de_liberdade,
                                                método=método if método is not None else self._método_padrão)

        f, u, ifc, iuc = self.incorporar_condições_de_contorno(malha,
                                                               graus_de_liberdade,
                                                               self.parâmetros_do_problema)

        # Lógica de determinação de f e u
        Kfc = self._onde_f_é_conhecido_fatiar(K, ifc)
        ufc = self._resolver_sistema_linear(Kfc, f, ifc)

        self._atualizar_graus_de_liberdade(u, ifc, ufc)

        Kuc = self._onde_u_é_conhecido_fatiar(K, iuc)
        self._atualizar_valores_de(f, iuc, Kuc, u)

        self._desligar_monitoramento()

        return f, u, malha

    def _configurar_monitoramento(self, monitorar: bool) -> None:
        """Se monitorar é True, liga o Monitorador e inicia seu timer"""
        if monitorar:
            self._monitoramento_ativo = True
            self._última_medição = self._início_do_monitoramento = default_timer()

    def _desligar_monitoramento(self) -> None:
        self._monitoramento_ativo = False
        self._última_medição = self._início_do_monitoramento = None

    @abstractmethod
    def determinar_graus_de_liberdade(self, malha: Malha) -> int:
        """Para a malha fornecida, retornará o número total de graus de liberdade do conjunto de seus nós."""

    @abstractmethod
    @Monitorador(mensagem="Matrizes de rigidez local determinadas")
    def calcular_matrizes_de_rigidez_local(self, **parâmetros_dos_elementos) -> Union[Matriz, Container[Matriz]]:
        """Retornará, a partir dos parâmetros fornecidos, as matrizes de rigidez local de cada elemento"""

    @abstractmethod
    @Monitorador(mensagem="Matriz de rigidez global montada")
    def montar_matriz_de_rigidez_geral(self,
                                       malha: Malha,
                                       Ks_locais: Union[Matriz, Container[Matriz]],
                                       graus_de_liberdade: int,
                                       método: str
                                       ) -> Matriz:
        """Montará a matriz de rigidez geral da malha usando o método especificado."""

    @abstractmethod
    @Monitorador(mensagem="Condições de contorno incorporadas")
    def incorporar_condições_de_contorno(self,
                                         malha: Malha,
                                         graus_de_liberdade: int,
                                         parâmetros_do_problema: Dict[str, Union[str, int, float]]
                                         ) -> Tuple[Vetor, Vetor, Máscara, Máscara]:
        """Atualizará os valores dos vetores f e u para incorporar as condições de contorno."""

    @Monitorador(mensagem="K fatiado onde f é conhecido")
    def _onde_f_é_conhecido_fatiar(self, K: Matriz, ifc: Máscara) -> Matriz:
        return K[np.ix_(ifc, ifc)]

    @Monitorador(mensagem="Sistema linear resolvido onde f é conhecido")
    def _resolver_sistema_linear(self, Kfc: Matriz, f: Vetor, ifc: Máscara) -> Vetor:
        return solve(Kfc, f[ifc])

    @Monitorador(mensagem="Graus de liberdade atualizados com o resultado da etapa anterior")
    def _atualizar_graus_de_liberdade(self, u: Vetor, ifc: Máscara, ufc: Vetor) -> None:
        u[ifc] = ufc

    @Monitorador(mensagem="K fatiado onde u é conhecido")
    def _onde_u_é_conhecido_fatiar(self, K: Matriz, iuc: Máscara) -> Matriz:
        return K[np.ix_(iuc, iuc)]

    @Monitorador(mensagem="Valores de f atualizados pela multiplicação de K por u")
    def _atualizar_valores_de(self, f: Vetor, iuc: Máscara, Kuc: Matriz, u: Vetor) -> None:
        f[iuc] = Kuc @ u[iuc]
