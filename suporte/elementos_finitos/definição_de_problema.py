from timeit import default_timer
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Tuple, Container, MutableSequence

import numpy as np
from numpy.linalg import solve

from suporte.elementos_finitos import Malha, Vetor, Matriz


Máscara = MutableSequence[bool]


class Problema(ABC):

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

    # Implementação da análise por elementos finitos
    def resolver_para(self,
                      parâmetros_dos_elementos: Dict[str, float],
                      malha: Malha,
                      método: str = None,
                      monitorar: bool = False
                      ) -> Tuple[Vetor, Vetor, Malha]:

        self.configurar_monitoramento(monitorar)

        graus_de_liberdade = self.determinar_graus_de_liberdade(malha)

        Ks_locais = self.calcular_matrizes_de_rigidez_local(**parâmetros_dos_elementos)
        K = self.montar_matriz_de_rigidez_geral(malha, Ks_locais, graus_de_liberdade,
                                                método=método if método is not None else self._método_padrão)

        f, u, ifc, iuc = self.incorporar_condições_de_contorno(malha,
                                                               graus_de_liberdade,
                                                               self.parâmetros_do_problema)

        # Lógica de determinação de f e u
        Kfc = self.onde_f_é_conhecido_fatiar(K, ifc)
        ufc = self.resolver_sistema_linear(Kfc, f, ifc)

        self.atualizar_graus_de_liberdade(u, ifc, ufc)

        Kuc = self.onde_u_é_conhecido_fatiar(K, iuc)
        self.atualizar_valores_de(f, iuc, Kuc, u)

        self.desligar_monitoramento()

        return f, u, malha

    def configurar_monitoramento(self, monitorar: bool) -> None:
        if monitorar:
            self._monitoramento_ativo = True
            self._última_medição = self._início_do_monitoramento = default_timer()

    def desligar_monitoramento(self) -> None:
        self._monitoramento_ativo = False
        self._última_medição = self._início_do_monitoramento = None

    @abstractmethod
    def determinar_graus_de_liberdade(self, malha: Malha) -> int:
        pass

    @abstractmethod
    @Monitorador(mensagem="Matrizes de rigidez local determinadas")
    def calcular_matrizes_de_rigidez_local(self, **parâmetros_dos_elementos
                                           ) -> Union[Matriz, Container[Matriz]]:
        pass

    @abstractmethod
    @Monitorador(mensagem="Matriz de rigidez global montada")
    def montar_matriz_de_rigidez_geral(self,
                                       malha: Malha,
                                       Ks_locais: Union[Matriz, Container[Matriz]],
                                       graus_de_liberdade: int,
                                       método: str
                                       ) -> Matriz:
        pass

    @abstractmethod
    @Monitorador(mensagem="Condições de contorno incorporadas")
    def incorporar_condições_de_contorno(self,
                                         malha: Malha,
                                         graus_de_liberdade: int,
                                         **parâmetros_do_problema
                                         ) -> Tuple[Vetor, Vetor, Máscara, Máscara]:
        pass

    @Monitorador(mensagem="K fatiado onde f é conhecido")
    def onde_f_é_conhecido_fatiar(self, K: Matriz, ifc: Máscara) -> Matriz:
        return K[np.ix_(ifc, ifc)]

    @Monitorador(mensagem="Sistema linear resolvido onde f é conhecido")
    def resolver_sistema_linear(self, Kfc: Matriz, f: Vetor, ifc: Máscara) -> Vetor:
        return solve(Kfc, f[ifc])

    @Monitorador(mensagem="Graus de liberdade atualizados com o resultado da etapa anterior")
    def atualizar_graus_de_liberdade(self, u: Vetor, ifc: Máscara, ufc: Vetor) -> None:
        u[ifc] = ufc

    @Monitorador(mensagem="K fatiado onde u é conhecido")
    def onde_u_é_conhecido_fatiar(self, K: Matriz, iuc: Máscara) -> Matriz:
        return K[np.ix_(iuc, iuc)]

    @Monitorador(mensagem="Valores de f atualizados pela multiplicação de K por u")
    def atualizar_valores_de(self, f: Vetor, iuc: Máscara, Kuc: Matriz, u: Vetor) -> None:
        f[iuc] = Kuc @ u[iuc]
