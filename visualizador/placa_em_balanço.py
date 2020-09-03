from itertools import chain as união_de

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
    conv, i_conv = calcular_convergência(amb)

    plt.suptitle("Índice de Convergência: {:.2f}%".format(100 * i_conv))
    plt.imshow(1 - conv, cmap="hot")
    plt.colorbar()
    plt.show()


def calcular_convergência(amb):
    genes = [proj.gene for proj in amb.população]
    conv = np.mean(genes, axis=0)
    idc = sum(_mconv(conv).flat)/len(conv.flat)
    return conv, idc


@np.vectorize
def _mconv(V):
    return 4*(V**2) - 4*V + 1


def mostrar_projeto(proj,  k=20, arquivo=None):
    fig, gráficos = plt.subplots(1, 2, figsize=(12, 4))

    gráfico_esquerdo = gráficos[0]
    gráfico_direito  = gráficos[1]

    plotar_gene(proj, gráfico_esquerdo)
    plotar_malha(proj, gráfico_direito, k=k)

    fig.suptitle(f"Projeto {proj.nome}", y=0.92, size="x-large", weight="demibold")

    fig.set_tight_layout(True)

    if arquivo is None:
        plt.show()
    else:
        fig.savefig(arquivo)


def plotar_gene(proj, gráfico):
    gráfico.imshow(~proj.gene, cmap="hot")
    gráfico.set_title(f"Material Genético do Projeto {proj.nome}")


def plotar_malha(proj, gráfico, k=20):
    assert k >= 0, f"k = {k}, mas k deve ser maior ou igual a 0"

    bordas, lados_internos = _definir_bordas(proj.malha)

    # A premissa aqui é que o vetor u tem a mesma ordem dos nós na lista malha.nós
    for lado in união_de(bordas, lados_internos):
        v0, v1 = tuple(lado)
        i0, i1 = proj.malha.índice_de[v0], proj.malha.índice_de[v1]
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
    gráfico.set_title(f"Malha de {proj.nome} sujeita à carga do problema")


def _definir_bordas(malha):
    bordas = set()
    lados_internos = set()

    for elemento in malha.elementos:
        elemento.traçar_bordas()
        for lado in elemento.bordas:
            if lado in bordas:
                lados_internos.add(lado)
                bordas.remove(lado)
            else:
                bordas.add(lado)

    return bordas, lados_internos


def mostrar_ambiente(amb, semente=0, k=20, arquivo=None):
    proj = amb.população[0]

    fig, gráficos = plt.subplots(2, 2, figsize=(12, 8))

    ((                gráfico_do_gene,     gráfico_da_malha_deformada),
     (gráfico_do_mapa_de_convergência, gráfico_da_deformação_em_cores)) = gráficos

    plotar_gene(proj, gráfico_do_gene)
    plotar_malha(proj, gráfico_da_malha_deformada, k=k)
    plotar_malha_com_cores(proj, gráfico_da_deformação_em_cores)
    plotar_mapa_de_convergência(amb, gráfico_do_mapa_de_convergência)

    fig.suptitle(f"Geração final do ambiente começado com semente {semente}\n"
                 f"Indivíduo mais bem adaptado: {proj.nome}", size="x-large", weight="demibold")
    fig.set_tight_layout(True)

    if arquivo is None:
        plt.show()
    else:
        fig.savefig(arquivo)


def plotar_malha_com_cores(proj, gráfico, k=0, paleta="magma"):
    quadro = np.full((720, 1440), -0.002)
    for e, elemento in enumerate(proj.malha.elementos):
        is_nós = proj.malha.me[::2, e] // 2

        xs_nós = [nó.x for nó in elemento.nós]
        ys_nós = [nó.y for nó in elemento.nós]

        dxs_nós = [proj.u[2*i] for i in is_nós]
        dys_nós = [proj.u[2*i + 1] for i in is_nós]

        xs_pontos = np.tile(np.linspace(xs_nós[0], xs_nós[1], 21), (21, 1))
        ys_pontos = np.tile(np.linspace(ys_nós[0], ys_nós[3], 21), (21, 1)).T

        qsis = np.tile(np.linspace(-1, 1, 21), (21, 1))
        etas = -qsis.T

        dxs_pontos = (dxs_nós[0] * (((1 - qsis)*(1 + etas))/4)
                      + dxs_nós[1] * (((1 + qsis)*(1 + etas))/4)
                      + dxs_nós[2] * (((1 + qsis)*(1 - etas))/4)
                      + dxs_nós[3] * (((1 - qsis)*(1 - etas))/4))
        dys_pontos = (dys_nós[0] * (((1 - qsis) * (1 + etas)) / 4)
                      + dys_nós[1] * (((1 + qsis) * (1 + etas)) / 4)
                      + dys_nós[2] * (((1 + qsis) * (1 - etas)) / 4)
                      + dys_nós[3] * (((1 - qsis) * (1 - etas)) / 4))

        d_pontos = np.sqrt(dxs_pontos**2 + dys_pontos**2)

        I = np.rint(720*(1 - (ys_pontos + k*dys_pontos))) - 1
        J = np.rint(720*(xs_pontos + k*dxs_pontos)) - 1

        quadro[I.astype(int).flatten(), J.astype(int).flatten()] = d_pontos.flatten()

    gráfico.set_title(f"Deformações em {proj.nome}")

    im = gráfico.imshow(quadro, cmap=paleta, interpolation="none")
    plt.colorbar(im, ax=gráfico)


def plotar_mapa_de_convergência(amb, gráfico):
    conv, i_conv = calcular_convergência(amb)

    im = gráfico.imshow(1 - conv, cmap="gray")
    gráfico.set_title(f"Mapa de convergência ({100*i_conv:.2f}%)")

