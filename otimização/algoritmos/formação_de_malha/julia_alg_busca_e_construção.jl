module AlgBC

import Base.show
import Base.hash

using Printf

export teste_determinar_fenótipo

struct Nó
    x::Float64
    y::Float64
    etiqueta::NTuple{2, Int}
end
    hash(nó::Nó) = hash(nó.etiqueta)
    function show(io::IO, nó::Nó) 
        print(io, "Nó(x=$(@sprintf("%.2f", nó.x)), y=$(@sprintf("%.2f", nó.y)), etiqueta=$(nó.etiqueta))")
    end


struct Elemento
    nós::Tuple{Vararg{Nó}}
end


struct Malha
    elementos::Tuple{Vararg{Elemento}}
    nós::Tuple{Vararg{Nó}}
    me::Matrix{UInt}
    ne::Int
    índice_de::Dict{Nó, Int}

    function Malha(elementos, nós, me)
        ne = length(elementos)
        índice_de = Dict(nó => i for (i, nó) in enumerate(nós) )
        return new(elementos, nós, me, ne, índice_de)
    end
end
    function show(m::Malha)
        saída = "["
        for elem in m.elementos
            nó = elem.nós[1]
            saída *= "Elemento com UL: $(nó.etiqueta), "
        end
        print(saída[1:end - 2] * "]")
    end


function determinar_fenótipo(gene)
    function adicionar_à_malha_o_elemento_em_(i, j)
        fenótipo[i, j] = true
        
        x = (j - 1)*l
        y = 1 - (i - 1)*l
        (ul, ur, dr, dl) = (Nó(    x,     y, (    i,     j)),
                            Nó(x + l,     y, (    i, j + 1)),
                            Nó(x + l, y - l, (i + 1, j + 1)),
                            Nó(    x, y - l, (i + 1,     j)))

        índices_globais_dos_cantos = []

        for nó ∈ (ul, ur, dr, dl) 
            if nó.etiqueta ∉ etiquetas_de_nós_já_construídos
                push!(nós, nó)

                índice_na_malha[nó.etiqueta] = length(etiquetas_de_nós_já_construídos)
                push!(índices_globais_dos_cantos, length(etiquetas_de_nós_já_construídos))

                push!(etiquetas_de_nós_já_construídos, nó.etiqueta)
            else
                push!(índices_globais_dos_cantos, índice_na_malha[nó.etiqueta])
            end
        end

        iul, iur, idr, idl = índices_globais_dos_cantos
        
        me = [me [2iul, 2iul + 1, 2iur, 2iur + 1, 2idr, 2idr + 1, 2idl, 2idl + 1]]

        if ul.etiqueta ∉ etiquetas_de_elementos_já_construídos
            push!(elementos, Elemento((ul, ur, dr, dl)))
            push!(etiquetas_de_elementos_já_construídos, ul.etiqueta)
        end

    end #function adicionar_à_malha_o_elemento_em_

    elementos = []
    nós = []
    me = Matrix{Union{UInt, Missing}}(missing, 8, 0)

    etiquetas_de_elementos_já_construídos = Set()
    etiquetas_de_nós_já_construídos = Set()
    índice_na_malha = Dict()

    fenótipo = zeros(Bool, size(gene))
    n, m = size(fenótipo)

    i = n÷2 + 1
    j = m

    l = 1/n

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
            setdiff!(possíves_ramificações, ((i, j, "esquerda"), (i, j, "direita")))
            adicionar_à_malha_o_elemento_em_(i, j)

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
            setdiff!(possíves_ramificações, ((i, j, "esquerda"), (i, j, "direita")))
            adicionar_à_malha_o_elemento_em_(i, j)

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

    return fenótipo, borda_alcançada, Tuple(elementos), Tuple(nós), me
end #determinar_fenótipo


function teste_determinar_fenótipo(checar=false)
    T = true
    f = false
    
    gene_teste = [[T T f f T T f f f f];
                  [f T f f T f f f f f];
                  [f T T T T T f f T T];
                  [T f f f f T T T T f];
                  [T T f f f f T f f f]]
    
    resultado, borda_alcançada, elementos, nós, me = determinar_fenótipo(gene_teste)

    if checar
        resultado_esperado = [[T T f f T T f f f f];
                              [f T f f T f f f f f];
                              [f T T T T T f f T T];
                              [f f f f f T T T T f];
                              [f f f f f f T f f f]]
        
        @assert borda_alcançada
        @assert all(resultado .≡ resultado_esperado)
        @assert length(elementos) ≡ sum(resultado_esperado)
        @assert length(nós) ≡ 38

        show(Malha(elementos, nós, me))
        println("\n")
        show(nós)
    end
end #teste_determinar_fenótipo

end #AlgBC