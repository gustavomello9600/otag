import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.dpi'] = 200


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


def mostrar_projeto(proj,  k=1, arquivo=None):
    fig, gráficos = plt.subplots(1, 2, figsize=(12, 4))

    gráfico_esquerdo = gráficos[0]
    gráfico_direito  = gráficos[1]

    plotar_malha(proj, gráfico_esquerdo, k=k)
    plotar_gene(proj, gráfico_direito)

    fig.suptitle(f"Projeto {proj.nome}", y=0.92,  size="x-large", weight="demibold")

    fig.set_tight_layout(True)

    if arquivo is None:
        plt.show()

    else:
        fig.savefig(arquivo)


def plotar_malha(proj, gráfico, k=1):
    assert k >= 0, f"k = {k}, mas k deve ser maior ou igual a 0"

    bordas, lados_internos = definir_bordas(proj.malha)

    # A premissa aqui é que o vetor u tem a mesma ordem dos nós na lista malha.nós
    for lado in união_de(bordas, lados_internos):
        v0, v1 = tuple(lado)
        i0, i1 = proj.malha.índice_de(v0), proj.malha.índice_de(v1)
        dx0, dx1 = proj.u[2*i0], proj.u[2*i1]
        dy0, dy1 = proj.u[2*i0 + 1], proj.u[2*i1 + 1]

        X, Y = [v0.x + k * dx0, v1.x + k * dx1], [v0.y + k * dy0, v1.y + k * dy1]
        if lado in bordas:
            gráfico.plot(X, Y, "k-")
        else:
            gráfico.plot(X, Y, "k--", lw=0.2)

    gráfico.axvline(x=0, c="black", lw="3")
    gráfico.set_xlim((-0.2, 2.2))
    gráfico.set_ylim((-0.1, 1.1))
    gráfico.set_aspect('equal')


def definir_bordas(malha):
    bordas = set()
    lados_internos = set()

    for elemento in malha.elementos:
        for lado in elemento.bordas:
            if lado in bordas:
                lados_internos.add(lado)
                bordas.remove(lado)
            else:
                bordas.add(lado)

    return bordas, lados_internos


def união_de(bordas, lados_internos):
    for borda in bordas:
        yield borda
    for lado_interno in lados_internos:
        yield lado_interno


def plotar_gene(proj, gráfico):
    gráfico.imshow(~proj.gene, cmap="hot")