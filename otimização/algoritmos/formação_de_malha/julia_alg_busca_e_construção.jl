module AlgBC

export teste_determinar_fenótipo

    function determinar_fenótipo(gene)
        fenótipo = zeros(Bool, size(gene))
        n, _ = size(fenótipo)

        i = n÷2 + 1
        j = 2n

        possíves_ramificações = Set()
        último_movimento = "esquerda"
        borda_alcançada = false
        buscando = true
        descida = true
        subida = false

        while buscando
            partida = (i, j)

            while descida
                if i ≠ n
                    abaixo = gene[i + 1, j]
                    descida = abaixo                
                else
                    descida = false
                end

                if (j ≠ 2n) & (último_movimento ≠ "esquerda")
                    direita = gene[i, j + 1]
                    if direita & ~fenótipo[i, j + 1]
                        push!(possíves_ramificações, (i, j + 1, "direita"))
                    end
                end

                if j ≡ 1
                    borda_alcançada = true
                elseif último_movimento ≠ "direita"
                    esquerda = gene[i, j - 1]
                    if esquerda & ~fenótipo[i, j - 1]
                        push!(possíves_ramificações, (i, j - 1, "esquerda"))
                    end
                end

                fenótipo[i, j] = true

                # Aqui entra a função de adicionar o elemento à malha
                setdiff!(possíves_ramificações, ((i, j, "esquerda"), (i, j, "direita")))

                if descida
                    i += 1
                    último_movimento = "baixo"
                else
                    i_p, j_p = partida
                    if i_p ≠ 1
                        subida = gene[i_p - 1, j_p]
                    else
                        subida = false
                    end

                    if subida
                        i = i_p - 1
                    end
                end
            end #descida

            while subida
                
                if i ≠ 1
                    acima = gene[i - 1, j]
                    subida = acima
                else
                    subida = false
                end

                if (j ≠ 2n) & (último_movimento ≠ "esquerda")
                    direita = gene[i, j + 1]
                    if direita & ~fenótipo[i, j + 1]
                        push!(possíves_ramificações, (i, j + 1, "direita"))
                    end
                end

                if j ≡ 1
                    borda_alcançada = true
                elseif último_movimento ≠ "direita"
                    esquerda = gene[i, j - 1]
                    if esquerda & ~fenótipo[i, j - 1]
                        push!(possíves_ramificações, (i, j - 1, "esquerda"))
                    end
                end

                fenótipo[i, j] = true

                # Aqui entra a função de adicionar o elemento à malha
                setdiff!(possíves_ramificações, ((i, j, "esquerda"), (i, j, "direita")))

                if subida
                    i -= 1
                    último_movimento = "cima"
                end

            end #subida
        
            if length(possíves_ramificações) > 0
                i, j, último_movimento = pop!(possíves_ramificações)
                descida = true
                subida = false
            else
                buscando = false
            end

        end #busca

        return fenótipo, borda_alcançada
    end #determinar_fenótipo


    function teste_determinar_fenótipo()
        T = true
        f = false
        
        gene_teste = [[T T f f T T f f f f];
                      [f T f f T f f f f f];
                      [f T T T T T f f T T];
                      [T f f f f T T T T f];
                      [T T f f f f T f f f]]
        
        resultado_esperado = [[T T f f T T f f f f];
                              [f T f f T f f f f f];
                              [f T T T T T f f T T];
                              [f f f f f T T T T f];
                              [f f f f f f T f f f]]
        
        
        resultado, borda_alcançada = determinar_fenótipo(gene_teste)
        @assert assertion_1 = all(resultado .≡ resultado_esperado)
        @assert assertion_2 = borda_alcançada 
    end #teste_determinar_fenótipo

end #AlgBC