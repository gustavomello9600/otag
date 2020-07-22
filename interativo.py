import pickle
import numpy as np
from pathlib import Path
from random import seed, getstate, setstate
from placa_em_balanço import População_de_Projetos
from matplotlib.pyplot import imshow, plot, colorbar, show, legend

semente = 0
info_gerações = []

def mudar_semente(sem):
    global semente

    seed(sem)
    np.random.seed(sem)
    semente = sem

def salvar_estado(pop):
    raiz = Path.cwd()
    pasta_da_semente = "semente_{}".format(semente)
    pasta_da_geração = "geração_{}".format(pop.geração)
    caminho          = raiz / "dados" / pasta_da_semente / pasta_da_geração

    caminho.mkdir(parents=True, exist_ok=True)

    backup_da_população     = "população.b"
    backup_do_estado_numpy  = "estado_do_numpy.b"
    backup_do_estado_random = "estado_do_random.b"

    with open(caminho / backup_da_população, "wb") as backup:
        pickle.dump(pop, backup)

    with open(caminho / backup_do_estado_numpy, "wb") as backup:
        pickle.dump(np.random.get_state(), backup)

    with open(caminho / backup_do_estado_random, "wb") as backup:
        pickle.dump(getstate(), backup)

def carregar_estado(semente=0, geração=1):
    raiz = Path.cwd()
    pasta_da_semente = "semente_{}".format(semente)
    pasta_da_geração = "geração_{}".format(pop.geração)
    caminho = raiz / "dados" / pasta_da_semente / pasta_da_geração

    backup_da_população = "população.b"
    backup_do_estado_numpy = "estado_do_numpy.b"
    backup_do_estado_random = "estado_do_random.b"

    try:
        with open(caminho / backup_da_população, "rb") as backup:
            pop = pickle.load(backup)

        with open(caminho / backup_do_estado_numpy, "rb") as backup:
            estado = pickle.load(backup)
            np.random.set_state(estado)

        with open(caminho / backup_do_estado_random, "rb") as backup:
            estado = pickle.load(backup)
            setstate(estado)

        print("> Estados dos geradores de números aleatórios redefinidos para quando o backup foi feito")

        return pop

    except:
        print("> Não há registros da geração {} começada com a semente {}".format(geração, semente))

def filtrar_informações(pop):
    global info_gerações
    for gen in pop.gerações:
        adpts = [ind.adaptação for ind in gen]
        info_gerações.append((np.max(adpts), np.mean(adpts), np.min(adpts)))
    pop.gerações.clear()

def mostrar_progresso(info):
   X = range(len(info))
   Ymax, Ymed, Ymin = zip(*info)
   plot(X, Ymax, "r--")
   plot(X, Ymed, "k-")
   plot(X, Ymin, "b--")
   legend(["Máxima", "Média", "Mínima"])
   show()

def mapa_de_convergência(pop):
    conv = sum([ind.gene for ind in pop.indivíduos])/100
    m_conv = np.vectorize(lambda x: 4*(x**2) - 4*x + 1)
    i_conv = sum(m_conv(conv).flat)/len(conv.flat)
    print("Índice de Convergência: {:.2f}%".format(100*i_conv))
    imshow(conv, cmap="hot")
    colorbar()
    show()

def mostrar_indivíduo(i, pop, tipo="malha", k=1):
    proj = pop.indivíduos[i]

    if tipo == "malha":
        proj.malha.plot(proj.u, k)

    elif tipo == "gene":
        imshow(~proj.gene, cmap="hot")
        colorbar()
        show()

def ciclo_de_(n, pop):
    pop.avançar_gerações(n)
    filtrar_informações(pop)
    salvar_estado(pop)

def execução_típica(n=100, pop=False, semente=0):
    retornar = False
    if not pop:
        retornar = True
        mudar_semente(semente)
        pop = População_de_Projetos()

    for k in range(pop.geração, pop.geração + n):
        pop.próxima_geração()

        if k % 10 == 0:
            filtrar_informações(pop)

        if k < 100 and k % 10 == 0:
            salvar_estado(pop)
        elif k % 100 == 0:
            salvar_estado(pop)

    if retornar:
        return pop

mudar_semente(semente)