import numba
import numpy as np


@numba.njit(cache=True)
def determinar_fenótipo(gene, n):
    l = 1/n

    gene_útil = []
    for i in range(n):
        linha = []
        for j in range(2*n):
            linha.append(False)
        gene_útil.append(linha)

    # Define a posição inicial do algoritmo de busca
    i = n // 2
    j = 2*n - 1

    """elementos = []
    nós = []
    me = []"""

    # Inicializa estruturas de dados auxiliares que ajudam a manter curso dos índices dos nós e elementos
    """etiquetas_de_elementos_já_construídos = set()
    etiquetas_de_nós_já_construídos = set()
    índice_na_malha = dict()"""

    # Inicializa parâmetros do algoritmo de busca
    possíveis_ramificações = set()
    último_movimento = "esquerda"
    borda_alcançada = False
    buscando = True
    descida = True
    subida = False

    # Junta as variáveis importantes para métodos auxiliares
    """contexto = (l, gene_útil, elementos, nós, me, índice_na_malha,
                etiquetas_de_nós_já_construídos, etiquetas_de_elementos_já_construídos)
    """
    # Executa o algoritmo de busca
    while buscando:

        # busca vertical
        partida = (i, j)

        # começar descida
        while descida:

            # consigo descer mais?
            if i != n - 1:
                abaixo = gene[i + 1][j]
                descida = abaixo
            else:
                descida = False

            # há ramificações possíveis aqui do lado?
            if j != 2 * n - 1 and último_movimento != "esquerda":
                direita = gene[i][j + 1]
                if direita and not gene_útil[i][j + 1]:
                    possíveis_ramificações.add((i, j + 1, "direita"))

            if j == 0:
                borda_alcançada = True
            elif último_movimento != "direita":
                esquerda = gene[i][j - 1]
                if esquerda and not gene_útil[i][j - 1]:
                    possíveis_ramificações.add((i, j - 1, "esquerda"))

            gene_útil[i][j] = True
            # adicionar_à_malha_o_elemento_em(i, j, contexto=contexto)
            possíveis_ramificações.discard((i, j, "esquerda"))
            possíveis_ramificações.discard((i, j, "direita"))

            # Decide se continua descendo ou se passa a subir
            if descida:
                i = i + 1
                último_movimento = "baixo"
            else:
                if partida[0] != 0:
                    subida = gene[partida[0] - 1][partida[1]]
                else:
                    subida = False

                if subida:
                    i = partida[0] - 1

        # começar subida
        while subida:

            # consigo subir mais?
            if i != 0:
                acima = gene[i - 1][j]
                subida = acima
            else:
                subida = False

            # há ramificações possíveis aqui do lado?
            if j != 2 * n - 1 and último_movimento != "esquerda":
                direita = gene[i][j + 1]
                if direita and not gene_útil[i][j + 1]:
                    possíveis_ramificações.add((i, j + 1, "direita"))

            if j == 0:
                borda_alcançada = True

            if j != 0 and último_movimento != "direita":
                esquerda = gene[i][j - 1]
                if esquerda and not gene_útil[i][j - 1]:
                    possíveis_ramificações.add((i, j - 1, "esquerda"))

            gene_útil[i][j] = True
            # adicionar_à_malha_o_elemento_em(i, j, contexto=contexto)
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

    # Transforma a lista de tuplas em matriz
    # me = inicializar_matriz_(me)

    return gene_útil, borda_alcançada #, elementos, nós, me


def teste_determinar_fenótipo():
    T = True
    f = False
        
    gene_teste = np.array([[T, T, f, f, T, T, f, f, f, f],
                           [f, T, f, f, T, f, f, f, f, f],
                           [f, T, T, T, T, T, f, f, T, T],
                           [T, f, f, f, f, T, T, T, T, f],
                           [T, T, f, f, f, f, T, f, f, f]], dtype=bool)
    
    resultado_esperado = np.array([[T, T, f, f, T, T, f, f, f, f],
                                   [f, T, f, f, T, f, f, f, f, f],
                                   [f, T, T, T, T, T, f, f, T, T],
                                   [f, f, f, f, f, T, T, T, T, f],
                                   [f, f, f, f, f, f, T, f, f, f]], dtype=bool)
    
    resultado, borda_alcançada = determinar_fenótipo(gene_teste, 5)
    assert np.all(resultado == resultado_esperado)
    assert borda_alcançada