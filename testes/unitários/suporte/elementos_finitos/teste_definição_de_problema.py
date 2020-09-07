import pytest

from suporte.elementos_finitos.definição_de_problema import *


@pytest.fixture
def problema_teste():
    class ProblemaTeste(Problema):
        def determinar_graus_de_liberdade(self, malha: Malha) -> int:
            return 10

        def calcular_matrizes_de_rigidez_local(self, **parâmetros_dos_elementos) -> Union[Matriz, Container[Matriz]]:
            return np.identity(8)

        def montar_matriz_de_rigidez_geral(self, malha: Malha, Ks_locais: Union[Matriz, Container[Matriz]],
                                           graus_de_liberdade: int, método: str) -> Matriz:
            K = np.identity(12)
            K[(2, 3, 4, 5), (2, 3, 4, 5)] = 2
            return K

        def incorporar_condições_de_contorno(self,
                                             malha: Malha,
                                             graus_de_liberdade: int,
                                             parâmetros_do_problema
                                             ) -> Tuple[Vetor, Vetor, Máscara, Máscara]:
            return (np.array([np.nan, np.nan, 0, 0, 0, 0, np.nan, np.nan, 0, -1, 0, 0]),
                    np.array([0, 0, np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan, np.nan, np.nan, np.nan]),
                    np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool),
                    np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=bool))

    return ProblemaTeste({"a": 0, "b": 0.0, "c": "string"})


def teste_resolver_para(problema_teste):
    f, u, malha = problema_teste.resolver_para({"str": 0.0}, "malha", "none", True)
    f, u, malha = problema_teste.resolver_para({"str": 0.0}, "malha", "none", False)

    assert np.all(f == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0]))
    assert np.all(u == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0]))
    assert malha == "malha"
