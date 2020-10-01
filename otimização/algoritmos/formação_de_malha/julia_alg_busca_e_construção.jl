x = true
_ = false

gene_teste = [[x x _ _ _ _ _ _ _ _];
              [_ x _ x x _ _ _ _ _];
              [_ x x x x x _ _ x x];
              [x _ _ _ _ x x x x _];
              [x x _ _ _ _ x _ _ _]]

resultado_esperado = [[x x _ _ _ _ _ _ _ _];
                      [_ x _ x x _ _ _ _ _];
                      [_ x x x x x _ _ x x];
                      [_ _ _ _ _ x x x x _];
                      [_ _ _ _ _ _ x _ _ _]]

function determinar_fenótipo(gene, l)
    gene_útil = zeros(Bool, size(gene))
end