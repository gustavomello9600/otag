import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.dpi'] = 300


def mostrar_progresso(info):
    X = range(len(info))
    Ymax, Ymed, Ymin = zip(*info)
    plt.plot(X, Ymax, "r--")
    plt.plot(X, Ymed, "k-")
    plt.plot(X, Ymin, "b--")
    plt.legend(["Máxima", "Média", "Mínima"])
    plt.show()


def mapa_de_convergência(amb):
    conv = sum([ind.gene for ind in amb.indivíduos]) / 100
    m_conv = np.vectorize(lambda x: 4 * (x ** 2) - 4 * x + 1)
    i_conv = sum(m_conv(conv).flat) / len(conv.flat)
    print("Índice de Convergência: {:.2f}%".format(100 * i_conv))
    plt.imshow(conv, cmap="hot")
    plt.colorbar()
    plt.show()


def mostrar_projeto(proj,  k=1000):
    fig, gráficos = plt.subplots(1, 2, figsize=(11, 4))

    gráfico_esquerdo = gráficos[0]
    gráfico_direito  = gráficos[1]

    plotar_malha(proj, gráfico_esquerdo, k=k)
    plotar_gene(proj, gráfico_direito)

    fig.suptitle(f"{proj.nome}")

    fig.set_tight_layout(True)
    fig.savefig("teste.png")


def plotar_malha(proj, gráfico, k=1):
    método_plot(proj.malha, gráfico, proj.u, k=k)


def plotar_gene(proj, gráfico):
    gráfico.imshow(~proj.gene, cmap="hot")


def método_plot(malha, gráfico, deslocamento=None, k=1):
    if not malha.bordas_traçadas:
        malha.traçar_bordas()

    if deslocamento is None:
        deslocamento = np.zeros(2 * len(malha.nós))

    for lado in malha.lados:
        i0, i1 = malha.índice_de(lado[0]), malha.índice_de(lado[1])
        dx0, dx1 = deslocamento[2 * i0], deslocamento[2 * i1]
        dy0, dy1 = deslocamento[2 * i0 + 1], deslocamento[2 * i1 + 1]

        X, Y = [lado[0].x + k * dx0, lado[1].x + k * dx1], [lado[0].y + k * dy0, lado[1].y + k * dy1]
        if lado in malha.bordas:
            gráfico.plot(X, Y, "k-")
        else:
            gráfico.plot(X, Y, "k--", lw=0.2)

    gráfico.axvline(x=0, c="black", lw="3")
    gráfico.set_xlim((-0.2, 2.2))
    gráfico.set_ylim((-0.1, 1.1))
    gráfico.set_aspect('equal')

def definir_bordas(malha):
