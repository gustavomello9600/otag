import numba
import numpy as np
from collections import namedtuple

Nó = namedtuple("Nó", "x y etiqueta".split())
Elemento = namedtuple("Elemento", ["nós"])
Malha = namedtuple("Malha", ["elementos", "nós", "me"])

@numba.njit(cache=True)
def determinar_fenótipo(gene):
    n, m = gene.shape

    fenótipo = np.full((n, m), False)

    # Define a posição inicial do algoritmo de busca
    i = n // 2
    j = 2*n - 1

    # Inicializa parâmetros do algoritmo de busca
    possíveis_ramificações = set()
    último_movimento = "esquerda"
    borda_alcançada = False
    buscando = True
    descida = True
    subida = False

    # Executa o algoritmo de busca
    while buscando:

        # busca vertical
        partida = (i, j)

        # começar descida
        while descida:

            # consigo descer mais?
            if i != n - 1:
                abaixo = gene[i + 1, j]
                descida = abaixo
            else:
                descida = False

            # há ramificações possíveis aqui do lado?
            if j != 2 * n - 1 and último_movimento != "esquerda":
                direita = gene[i, j + 1]
                if direita and not fenótipo[i, j + 1]:
                    possíveis_ramificações.add((i, j + 1, "direita"))

            if j == 0:
                borda_alcançada = True
            elif último_movimento != "direita":
                esquerda = gene[i, j - 1]
                if esquerda and not fenótipo[i, j - 1]:
                    possíveis_ramificações.add((i, j - 1, "esquerda"))

            fenótipo[i, j] = True
            possíveis_ramificações.discard((i, j, "esquerda"))
            possíveis_ramificações.discard((i, j, "direita"))

            # Decide se continua descendo ou se passa a subir
            if descida:
                i = i + 1
                último_movimento = "baixo"
            else:
                if partida[0] != 0:
                    subida = gene[partida[0] - 1, partida[1]]
                else:
                    subida = False

                if subida:
                    i = partida[0] - 1

        # começar subida
        while subida:

            # consigo subir mais?
            if i != 0:
                acima = gene[i - 1, j]
                subida = acima
            else:
                subida = False

            # há ramificações possíveis aqui do lado?
            if j != 2 * n - 1 and último_movimento != "esquerda":
                direita = gene[i, j + 1]
                if direita and not fenótipo[i, j + 1]:
                    possíveis_ramificações.add((i, j + 1, "direita"))

            if j == 0:
                borda_alcançada = True

            if j != 0 and último_movimento != "direita":
                esquerda = gene[i, j - 1]
                if esquerda and not fenótipo[i, j - 1]:
                    possíveis_ramificações.add((i, j - 1, "esquerda"))

            fenótipo[i, j] = True
            possíveis_ramificações.discard((i, j, "esquerda"))
            possíveis_ramificações.discard((i, j, "direita"))

            # Decide se continua descendo ou se passa a subir
            if subida:
                i = i - 1
                último_movimento = "cima"

        if len(possíveis_ramificações) > 0:
            i, j, último_movimento = possíveis_ramificações.pop()
            descida = True
            subida = False

        else:
            buscando = False

    return fenótipo, borda_alcançada


def determinar_fenótipo_sem_compilar(gene):
    def adicionar_à_malha_o_elemento_em(i: int, j: int) -> None:
        # Inicializa os nós dos cantos do elemento
        y = 1 - i*l
        ul, ur, dr, dl = Nó(      j*l,     y, etiqueta=(    i,     j)), \
                         Nó((j + 1)*l,     y, etiqueta=(    i, j + 1)), \
                         Nó((j + 1)*l, y - l, etiqueta=(i + 1, j + 1)), \
                         Nó(      j*l, y - l, etiqueta=(i + 1,     j))

        índices_globais_dos_cantos = []

        # Para cada nó em cada canto
        for nó in (ul, ur, dr, dl):

            # Se o índice do nó ainda não foi visto
            if nó.etiqueta not in etiquetas_de_nós_já_construídos:

                # Adiciona o canto à lista de nós que comporão a malha
                nós.append(nó)

                # Define o índice do canto na malha como o índice corrente
                índice_na_malha[nó.etiqueta] = len(etiquetas_de_nós_já_construídos)
                índices_globais_dos_cantos.append(len(etiquetas_de_nós_já_construídos))

                # Define que o nó já foi visto
                etiquetas_de_nós_já_construídos.add(nó.etiqueta)

            else:
                índices_globais_dos_cantos.append(índice_na_malha[nó.etiqueta])

        iul, iur, idr, idl = índices_globais_dos_cantos

        # Atualiza a matriz de correspondência entre os índices
        # locais e globais dos nós para o elemento atual
        me.append((2 * iul,
                   2 * iul + 1,
                   2 * iur,
                   2 * iur + 1,
                   2 * idr,
                   2 * idr + 1,
                   2 * idl,
                   2 * idl + 1))

        # Cria um elemento com os cantos e o adiciona
        # à lista daqueles que comporão a malha
        if ul.etiqueta not in etiquetas_de_elementos_já_construídos:
            elementos.append(Elemento((ul, ur, dr, dl)))
            etiquetas_de_elementos_já_construídos.add(ul.etiqueta)

    n, m = gene.shape
    l = 1/n

    fenótipo = np.zeros((n, m), dtype=bool)

    # Define a posição inicial do algoritmo de busca
    i = n // 2
    j = 2*n - 1

    elementos = []
    nós = []
    me = []

    # Inicializa estruturas de dados auxiliares que ajudam a manter curso dos índices dos nós e elementos
    etiquetas_de_elementos_já_construídos = set()
    etiquetas_de_nós_já_construídos = set()
    índice_na_malha = dict()

    # Inicializa parâmetros do algoritmo de busca
    possíveis_ramificações = set()
    último_movimento = "esquerda"
    borda_alcançada = False
    buscando = True
    descida = True
    subida = False

    # Executa o algoritmo de busca
    while buscando:

        # busca vertical
        partida = (i, j)

        # começar descida
        while descida:

            # consigo descer mais?
            if i != n - 1:
                abaixo = gene[i + 1, j]
                descida = abaixo
            else:
                descida = False

            # há ramificações possíveis aqui do lado?
            if j != 2 * n - 1 and último_movimento != "esquerda":
                direita = gene[i, j + 1]
                if direita and not fenótipo[i, j + 1]:
                    possíveis_ramificações.add((i, j + 1, "direita"))

            if j == 0:
                borda_alcançada = True
            elif último_movimento != "direita":
                esquerda = gene[i, j - 1]
                if esquerda and not fenótipo[i, j - 1]:
                    possíveis_ramificações.add((i, j - 1, "esquerda"))

            fenótipo[i, j] = True
            adicionar_à_malha_o_elemento_em(i, j)
            possíveis_ramificações.discard((i, j, "esquerda"))
            possíveis_ramificações.discard((i, j, "direita"))

            # Decide se continua descendo ou se passa a subir
            if descida:
                i = i + 1
                último_movimento = "baixo"
            else:
                if partida[0] != 0:
                    subida = gene[partida[0] - 1, partida[1]]
                else:
                    subida = False

                if subida:
                    i = partida[0] - 1

        # começar subida
        while subida:

            # consigo subir mais?
            if i != 0:
                acima = gene[i - 1, j]
                subida = acima
            else:
                subida = False

            # há ramificações possíveis aqui do lado?
            if j != 2 * n - 1 and último_movimento != "esquerda":
                direita = gene[i, j + 1]
                if direita and not fenótipo[i, j + 1]:
                    possíveis_ramificações.add((i, j + 1, "direita"))

            if j == 0:
                borda_alcançada = True

            if j != 0 and último_movimento != "direita":
                esquerda = gene[i, j - 1]
                if esquerda and not fenótipo[i, j - 1]:
                    possíveis_ramificações.add((i, j - 1, "esquerda"))

            fenótipo[i, j] = True
            adicionar_à_malha_o_elemento_em(i, j)
            possíveis_ramificações.discard((i, j, "esquerda"))
            possíveis_ramificações.discard((i, j, "direita"))

            # Decide se continua descendo ou se passa a subir
            if subida:
                i = i - 1
                último_movimento = "cima"

        if len(possíveis_ramificações) > 0:
            i, j, último_movimento = possíveis_ramificações.pop()
            descida = True
            subida = False

        else:
            buscando = False

    return fenótipo, borda_alcançada, elementos, nós, me


def teste_determinar_fenótipo(compilar=True, checar=False):
    T = True
    f = False
        
    gene_teste = np.array([[T, T, f, f, T, T, f, f, f, f],
                           [f, T, f, f, T, f, f, f, f, f],
                           [f, T, T, T, T, T, f, f, T, T],
                           [T, f, f, f, f, T, T, T, T, f],
                           [T, T, f, f, f, f, T, f, f, f]], dtype=bool)
    
    if compilar:
        resultado, borda_alcançada = determinar_fenótipo(gene_teste)
    else:
        resultado, borda_alcançada, elementos, nós, me = determinar_fenótipo_sem_compilar(gene_teste)

    if checar:
        resultado_esperado = np.array([[T, T, f, f, T, T, f, f, f, f],
                                       [f, T, f, f, T, f, f, f, f, f],
                                       [f, T, T, T, T, T, f, f, T, T],
                                       [f, f, f, f, f, T, T, T, T, f],
                                       [f, f, f, f, f, f, T, f, f, f]], dtype=bool)
        assert np.all(resultado == resultado_esperado)
        assert borda_alcançada

        if not compilar:
            assert len(elementos) == resultado_esperado.sum()
            assert len(nós) == 38

            malha = Malha(elementos, nós, me)
            for at in malha:
                print(at)
                print("\n")