# -*- coding: UTF-8 -*-

# Importa o Numerical Python
import numpy as np

# Importa as ferramentas de plotagem do Matplotlib
import matplotlib.pyplot as plt


# a) Primeira implementação do operador de crossover
def crossover_a(pai_1, pai_2):

    # Escolhe a posição da reta vertical aleatoriamente
    j = np.random.choice(range(1, nelx))

    # Forma o material genético do filho 1 de acordo com o operador
    filho_1 = np.concatenate((pai_1[:, :j], pai_2[:, j:]), axis=1)

    # Forma o material genético do filho 2 como a versão invertida do filho 1
    filho_2 = ~filho_1

    # Retorna os indivíduos filhos
    return filho_1, filho_2

# b) Segunda implementação do operador de crossover
def crossover_b(pai_1, pai_2):

    # Escolhe um ponto de partida aleatório para a reta diagonal
    i_0 = np.random.choice(range(1, nely))
    j_0 = np.random.choice(range(1, nelx))

    # Escolhe uma inclinação aleatória para a reta diagonal
    alfa = 2* np.pi * np.random.random()

    # Retorna um valor real j para cada ponto com índice i pertencente à reta
    reta = lambda i: j_0 - (i - i_0)*np.tan(alfa)

    # Forma o material genético do filho 1 de acordo com o operador
    filho_1 = np.array([[pai_1[i, j] if reta(i) > j else pai_2[i, j]
                                     for j in range(nelx)]
                                     for i in range(nely)])

    # Forma o material genético do filho 2 como a versão invertida do filho 1
    filho_2 = ~filho_1

    # Retorna os indivíduos filhos
    return filho_1, filho_2


# Define o número de elementos_finitos da malha em cada direção
nelx = 120
nely = 40

# Inicia o genótipo de cada indivíduo pai
pai_1 = np.ones( (nely, nelx), dtype=bool)
pai_2 = np.zeros((nely, nelx), dtype=bool)

# Ensina o Python como mostrar o resultado das operações de crossover
def comparar_pais_e_filhos(pai_1, pai_2, filho_1, filho_2, titulo="Crossover"):

    # Inicia uma imagem com 4 subgráficos dispostos em 2 linhas e 2 colunas
    # e com proporção horizontal/vertical de 9 para 5
    fig, axs = plt.subplots(2, 2, figsize=(9, 5))

    # Para cada indivíduo, representa e identifica seu genótipo em um dos subgráficos
    axs[0, 0].imshow(pai_1, vmin=0, vmax=1)
    axs[0, 0].set_title("Pai 1")

    axs[1, 0].imshow(pai_2, vmin=0, vmax=1)
    axs[1, 0].set_title("Pai 2")

    axs[0, 1].imshow(filho_1, vmin=0, vmax=1)
    axs[0, 1].set_title("Filho 1")

    axs[1, 1].imshow(filho_2, vmin=0, vmax=1)
    axs[1, 1].set_title("Filho 2")

    # Escreve o título da imagem
    fig.suptitle(titulo)

    # Mostra a imagem
    plt.show()


# Executa o operador de crossover a)
filho_1a, filho_2a = crossover_a(pai_1, pai_2)

# Mostra o seu resultado
comparar_pais_e_filhos(pai_1=pai_1, pai_2=pai_2, filho_1=filho_1a, filho_2=filho_2a,
                       titulo="Crossover A (reta vertical)")


# Executa o operador de crossover b)
filho_1b, filho_2b = crossover_b(pai_1, pai_2)

# Mostra o seu resultado
comparar_pais_e_filhos(pai_1=pai_1, pai_2=pai_2, filho_1=filho_1b, filho_2=filho_2b,
                       titulo="Crossover B (reta diagonal)")

"""
OBS: Esse código foi formatado utilizando as recomendações do PEP (Python Enhancement Package) 8
facilmente acessível através do endereço python.org
"""