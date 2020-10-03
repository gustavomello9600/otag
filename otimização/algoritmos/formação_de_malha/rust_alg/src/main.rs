#[macro_use]
extern crate timeit;
extern crate ndarray;

use ndarray::prelude::*;
use std::collections::HashSet;

fn main() {
    timeit!({
        teste_determinar_fenotipo();
    });
}

fn determinar_fenotipo(gene: &Array2<bool>) -> (Array2<bool>, bool) {
    let (n, m) = gene.dim();
    let mut gene_util = Array::from_elem((n, m), false);

    let mut i = n / 2;
    let mut j = m - 1;

    let mut possiveis_ramificacoes: HashSet<(usize, usize, Movimento)> = HashSet::new();
    let mut ultimo_movimento = Movimento::Esquerda;
    let mut borda_alcancada = false;
    let mut buscando = true;
    let mut descida = true;
    let mut subida = false;

    while buscando {
        let partida = (i, j);

        while descida {
            if i != n - 1 {
                descida = gene[[i + 1, j]];
            } else {
                descida = false;
            }

            if (j != 2*n - 1) & (ultimo_movimento != Movimento::Esquerda) {
                let direita = gene[[i, j + 1]];
                if (direita) & !(gene_util[[i, j + 1]]) {
                    possiveis_ramificacoes.insert( (i, j + 1, Movimento::Direita) );
                }
            }

            if j == 0 {
                borda_alcancada = true;
            } else if ultimo_movimento != Movimento::Direita {
                let esquerda = gene[[i, j - 1]];
                if esquerda & !(gene_util[[i, j - 1]]) {
                    possiveis_ramificacoes.insert( (i, j - 1, Movimento::Esquerda) );
                }
            }

            gene_util[[i, j]] = true;
            // adicionar_a_malha_o_elemento_em_(i, j);
            possiveis_ramificacoes.remove(&(i, j, Movimento::Esquerda));
            possiveis_ramificacoes.remove(&(i, j, Movimento::Direita));

            if descida {
                i += 1;
                ultimo_movimento = Movimento::Baixo;
            } else {
                if partida.0 != 0 {
                    subida = gene[[partida.0 - 1, partida.1]];
                } else {
                    subida = false
                }

                if subida {
                    i = partida.0 - 1;
                }
            }
        }

        while subida {
            if i != 0 {
                subida = gene[[i - 1, j]];
            } else {
                subida = false
            }

            if (j != 2*n - 1) & (ultimo_movimento != Movimento::Esquerda) {
                let direita = gene[[i, j + 1]];
                if direita & !gene_util[[i, j + 1]] {
                    possiveis_ramificacoes.insert( (i, j + 1, Movimento::Direita) );
                }
            }

            if j == 0 {
                borda_alcancada = true;
            } else if ultimo_movimento != Movimento::Direita {
                let esquerda = gene[[i, j - 1]];
                if (esquerda) & !(gene_util[[i, j - 1]]) {
                    possiveis_ramificacoes.insert( (i, j - 1, Movimento::Esquerda) );
                }
            }

            gene_util[[i, j]] = true;
            // adicionar_a_malha_o_elemento_em_(i, j);
            possiveis_ramificacoes.remove(&(i, j, Movimento::Esquerda));
            possiveis_ramificacoes.remove(&(i, j, Movimento::Direita));

            if subida {
                i -= 1;
                ultimo_movimento = Movimento::Cima;
            }
        }

        if possiveis_ramificacoes.len() > 0 {
            let elemento = possiveis_ramificacoes.iter().next().cloned().unwrap();
            let elemento = possiveis_ramificacoes.take(&elemento).unwrap();
            i = elemento.0;
            j = elemento.1;
            ultimo_movimento = elemento.2;

            descida = true;
            subida = false;
        } else {
            buscando = false;
        }
    }

    return (gene_util, borda_alcancada);
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
enum Movimento {
    Direita,
    Esquerda,
    Cima,
    Baixo
}

fn teste_determinar_fenotipo() {
    let (t, f) = (true, false);

    let gene_teste = array![[t, t, f, f, t, t, f, f, f, f],
                            [f, t, f, f, t, f, f, f, f, f],
                            [f, t, t, t, t, t, f, f, t, t],
                            [t, f, f, f, f, t, t, t, t, f],
                            [t, t, f, f, f, f, t, f, f, f]];

    let resultado_esperado = array![[t, t, f, f, t, t, f, f, f, f],
                                    [f, t, f, f, t, f, f, f, f, f],
                                    [f, t, t, t, t, t, f, f, t, t],
                                    [f, f, f, f, f, t, t, t, t, f],
                                    [f, f, f, f, f, f, t, f, f, f]];

    let (resultado, borda_alcancada) = determinar_fenotipo(&gene_teste);
    assert!(resultado == resultado_esperado);
    assert!(borda_alcancada);
}
