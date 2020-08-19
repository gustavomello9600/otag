from suporte.elementos.membrana_quadrada import *
from suporte.elementos_finitos import Malha


def criar_malha_cheia(n):
    l = 1/n

    nós       = []
    elementos = []
    me        = np.zeros((8, 2*(n**2)), dtype="int16" )

    e = 0
    for i in range(n, 0, -1):
        for j in range(2 * n):
            if i == n:
                if j == 0:
                    ul = Nó((      j*l,       i*l))
                    ur = Nó(((j + 1)*l,       i*l))
                    dr = Nó(((j + 1)*l, (i - 1)*l))
                    dl = Nó((      j*l, (i - 1)*l))
                    nós.extend([ul, ur, dr, dl])
                else:
                    ul = nós[-2 - (0 if j != 1 else 1)]
                    ur = Nó(((j + 1)*l,       i*l))
                    dr = Nó(((j + 1)*l, (i - 1)*l))
                    dl = nós[-1 - (0 if j != 1 else 1)]
                    nós.extend([ur, dr])

            else:
                if j == 0:
                    ul = nós[3 if i == n - 1 else (n - i)*(2*n + 1) + 1]
                    ur = nós[2 if i == n - 1 else (n - i)*(2*n + 1)]
                    dr = Nó(((j + 1)*l, (i - 1)*l))
                    dl = Nó((      j*l, (i - 1)*l))
                    nós.extend([dr, dl])
                else:
                    if i == n - 1:
                        ul = nós[2 if j == 1 else 2*j + 1]
                        ur = nós[2*(j+1) + 1]
                        dr = Nó(((j + 1) * l, (i - 1) * l))
                        dl = nós[-2 if j == 1 else -1]
                        nós.extend([dr])
                    else:
                        ul = nós[-2*(n + 1) if j != 1 else -2*(n + 1) - 1]
                        ur = nós[-2*(n + 1) + 1]
                        dr = Nó(((j + 1) * l, (i - 1) * l))
                        dl = nós[-1 if j != 1 else -2]
                        nós.extend([dr])

            elementos.append(MembranaQuadrada([ul, ur, dr, dl]))
            ie, je   = e // (2*n), e % (2*n)
            me[:, e] = 2*np.array([[          ie*(2*n + 1) + je,   ie*(2*n + 1) + je + 1,
                                    (ie + 1)*(2*n + 1) + je + 1, (ie + 1)*(2*n + 1) + je]], dtype="int16"
                                  ).repeat(2) + np.array([0, 1, 0, 1, 0, 1, 0, 1])
            e += 1

    nós = sorted(nós, reverse=True)

    return Malha(elementos, nós, me)


assert len(criar_malha_cheia(38).elementos) == 38*75, f"O resultado deveria ser {38*76}"