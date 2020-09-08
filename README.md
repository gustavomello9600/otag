# Design Generativo com Python
![versão_do_python](https://img.shields.io/badge/python-3.7%2B-green)
![arquitetura_master](https://img.shields.io/badge/arquitetura-v1.1-blue)
![melhorias_planejadas](https://img.shields.io/github/issues-raw/gustavomello9600/otag?label=melhorias%20planejadas)
![testagem_dos_módulos_de_suporte](https://img.shields.io/badge/testagem%20dos%20m%C3%B3dulos%20de%20suporte-100%25-brightgreen)

<p align="center">
  <img src="https://i.imgur.com/JR4uMyO.png" />
</p>

Design Generativo com Python é uma biblioteca que implementa abstrações capazes de otimizar projetos de estruturas simples através de algoritmos genéticos.

<p align="center">
  <img src="https://media1.giphy.com/media/XElPH6AihVmOHYx5yk/giphy.gif" />
</p>

## Principais funcionalidades
* Arquitetura Modular que permite o uso de um mesmo conjunto de operadores genéticos para diferentes problemas e vice-versa;
* Parâmetros são fornecidos através de um arquivo JSON facilmente entendível e editável por humanos;
* Algoritmos de análise por elementos finitos otimizados e paralelizados;
* Código dos módulos de suporte 100% cobertos por testes.

## Sumário
  * [Versões da Arquitetura](#vers-es-da-arquitetura)
    + [Arquitetura v1.0](#arquitetura-v10)
      - [Interpretador](#interpretador)
      - [Algoritmo Genético](#algoritmo-gen-tico)
      - [Modelo de Discretização](#modelo-de-discretiza--o)
      - [Definição de Problema](#defini--o-de-problema)
      - [Situação de Projeto](#situa--o-de-projeto)
      - [Fachada](#fachada)
      - [Visualizador](#visualizador)
    + [Arquitetura v1.1](#arquitetura-v11)
      - [Parâmetros, Ambiente de Projeto e Problema Específico](#par-metros--ambiente-de-projeto-e-problema-espec-fico)
      - [Estrutura de pastas](#estrutura-de-pastas)
  * [Como Usar](#como-usar)
  * [Sobre o Autor](#sobre-o-autor)

## Versões da Arquitetura

### Arquitetura v1.0
*branch: origin/v1.0*
![Maquetev1.0](https://i.imgur.com/ixCfjAy.png)

De baixo para cima, a responsabilidade de cada camada evolui gradualmente desde a comunicação de instruções para as unidades de processamento até
o gerenciamento dos comandos fornecidos pelo usuário. As camadas superiores se sustentam nas abstrações das camadas inferiores.

#### Interpretador
Opera o intérprete do Python em versão igual ou superior a 3.7 com o suporte das seguintes bibliotecas, ordenadas a partir das mais relevantes
para a execução:

* **numpy**. Biblioteca de cálculo numérico presente na maior parte dos módulos implementando funcionalidades essenciais;
* **sympy**. Biblioteca de álgebra computacional empregada principalmente na construção de matrizes de rigidez local das implementações de
elementos finitos;
* **matplotlib**. Biblioteca de desenho de gráficos utilizada extensivamente pela camada de visualização;
* **more-itertools**. Biblioteca de ferramentas de manipulação de iteráveis que otimiza algumas funções da camada de visualização;
* **pandas**. Python Data Analysis - Biblioteca de análise de dados empregada pela camada de execução para produzir e atualizar a planilha de
comparação de resultados.

* **pytest**. (Opcional) - Framework de testagem que permite a automatização dos testes unitários dos módulos de suporte.

#### Algoritmo Genético
Define as classes Ambiente e Indivíduo que implementam operadores genéticos e estruturas de dados que carregam genes.

#### Modelo de Discretização
Define as classes Malha, Elemento, Nó e KeBase utilizadas nas análises por elementos finitos. Por cima desta camada são construídas as implementações
de cada tipo específico de elemento finito.

#### Definição de Problema
Implementa um método base de resolução das malhas de elementos finitos. Incorpora ainda um Monitorador que registra o tempo de execução de cada passo.

#### Situação de Projeto
Camada modular da arquitetura. É fornecida pelo usuário e contém uma implementação concreta de operadores genéticos através da definição de uma classe
AmbienteDeProjeto que recebe, por composição, uma classe SituaçãoDeProjeto que consome parâmetros fornecidos no início do arquivo e define um problema
concreto.

#### Fachada
Interface simplificada através da qual o usuário opera o sistema. É responsável por chamar a camada de visualização e salvar resultados e dados
de execução.

#### Visualizador
Traz ferramentas para traduzir para gráficos os objetos definidos no Modelo de Discretização.

### Arquitetura v1.1
*branch: origin/master*
![Maquetev1.1](https://i.imgur.com/k2hK0W9.png)

#### Parâmetros, Ambiente de Projeto e Problema Específico
As três componentes que eram uma única camada (Situação de Projeto) na arquitetura v1.0 foram divididas para permitir a execução de um mesmo conjunto
de operadores genéticos para uma dada definição de problema, por exemplo. Os parâmetros da execução são definidos em arquivos JSON facilmente editáveis.

#### Estrutura de pastas
* A pasta **otimização** contém relatórios e testes dos algoritmos executados em diferentes implementações e com diferentes backends.
* A pasta **situações_de_projeto** é onde estão localizados os módulos criados pelo usuário para modelagem da situação de projeto
e os resultados/dados de suas execuções para cada combinação Ambiente + Projeto + Parâmetros.
* A pasta **suporte** contém os arquivos das camadas inferiores da arquitetura.
* A pasta **teste** contém os diferentes tipos de testes que validam o bom funcionamento do sistema.
* A pasta **visualizador** contém um módulo específico para a visualização de cada situação de projeto.

```bash
$ tree
.
├── otimização
│   ├── algoritmos
│   └── backend
├── situações_de_projeto
│   ├── situação_exemplo
│   │   ├── ambientes
│   │   │   ├── __init__.py
│   │   │   └── ambiente_exemplo.py
│   │   ├── dados
│   │   │   └── ambiente_exemplo_problema_exemplo_parâmatros_exemplo
│   │   │       ├── semente_0
│   │   │       └── ...
│   │   ├── parâmetros
│   │   │   ├── padrão.json
│   │   │   └── parâmetros_exemplo.json
│   │   ├── problemas
│   │   │   ├── __init__.py
│   │   │   └── problema_exemplo.py
│   │   ├── resultados
│   │   │   └── ambiente_exemplo_problema_exemplo_parâmatros_exemplo
│   │   │       ├── comparação_de_resultados.csv
│   │   │       ├── semente_0.png
│   │   │       └── ...
│   │   └── __init__.py
│   └── __init__.py
├── suporte
│   ├── algoritmo_genético.py
│   ├── elementos_finitos
│   │   ├── cache
│   │   │   └── K_elemento_específico_base.b
│   │   ├── __init__.py
│   │   ├── definição_de_problema.py
│   │   └── elemento_específico.py
│   └── __init__.py
├── testes
│   ├── unitários
│   ├── integração
│   ├── ponta_a_ponta
│   └── __init__.py
├── visualizador
│   ├── __init__.py
│   └── situação_exemplo.py
└── otag.py
```

## Como Usar

1. Crie uma pasta para a situação de projeto e três subpastas: **ambientes**, **problemas** e **parâmetros**.
```
.
├── situações_de_projeto
│   ├── situação_exemplo
│   │   ├── ambientes
│   │   │   ├── __init__.py
│   │   │   └── ambiente_exemplo.py
│   │   ├── parâmetros
│   │   │   └── padrão.json
│   │   ├── problemas
│   │   │   ├── __init__.py
│   │   │   └── problema_exemplo.py
```
2. Dentro da pasta **ambientes**, crie um arquivo como `ambiente_exemplo.py` e implemente a classe `AmbienteDeProjeto`
com no mínimo os mesmos métodos.
```python3
# ambiente_exemplo.py
from tipyng import List

from suporte.algoritmo_genético import Ambiente, Indivíduo


class AmbienteDeProjeto(Ambiente):
    def geração_0(self) -> List[Indivíduo]:
    """Construirá a geração inicial da população do ambiente."""
    
    def testar_adaptação(self, indivíduo: Indivíduo) -> None:
    """Testará a adaptação do indivíduo, modificando seus atributos."""
    
    def crossover(self, pai1: Indivíduo, pai2: Indivíduo, i: int) -> Indivíduo:
    """Gerará um novo indivíduo a partir do cruzamento dos genes de seus pais."""
    
    def mutação(self, geração: List[Indivíduo]) -> None:
    """Modificará a nova geração de acordo com as regras estabelecidas pelo ambiente."""

```
3. Dentro da pasta **problemas**, crie um arquivo como `problema_exemplo.py` e implemente uma classe como `ExemploDeProblema`
com no mínimo os mesmos métodos.
```python3
# problema_exemplo.py
from typing import Dict, Union, Container

from suporte.elementos_finitos import Malha, Vetor, Matriz
from suporte.elementos_finitos.definição_de_problema import Problema, Máscara


class ExemploDeProblema(Problema):
    def determinar_graus_de_liberdade(self, malha: Malha) -> int:
    """Para a malha fornecida, retornará o número total de graus de liberdade do conjunto de seus nós."""
    
    def calcular_matrizes_de_rigidez_local(self, **parâmetros_dos_elementos) -> Union[Matriz, Container[Matriz]]:
    """Retornará, a partir dos parâmetros fornecidos, as matrizes de rigidez local de cada elemento"""
    
    def montar_matriz_de_rigidez_geral(self,
                                       malha: Malha,
                                       Ks_locais: Union[Matriz, Container[Matriz]],
                                       graus_de_liberdade: int,
                                       método: str
                                       ) -> Matriz:
    """Montará a matriz de rigidez geral da malha usando o método especificado."""
    
    def incorporar_condições_de_contorno(self,
                                         malha: Malha,
                                         graus_de_liberdade: int,
                                         parâmetros_do_problema: Dict[str, Union[str, int, float]]
                                         ) -> Tuple[Vetor, Vetor, Máscara, Máscara]:
    """Atualizará os valores dos vetores f e u para incorporar as condições de contorno."""

```
4. Defina os parâmetros que serão consumidos pelas instâncias da classe `ExemploDeProblema` e os armazene na pasta
**parâmetros** num arquivo como `padrão.json`.
```JSON
{
  "EXEMPLO_DE_CONSTANTE_DO_MATERIAL": "0.5",
  "EXEMPLO_DE_PARÂMETRO_DO_PROBLEMA": "10",
  "EXEMPLO_DE_MÉTODO_PADRÃO": "Nome do Método"
}
```
5. Abra uma sessão interativa do Python na pasta raiz do repositório e importe o módulo `otag`.
6. Chame a função `otag.interativo(gerações=20)` com o número de gerações desejado.
7. Escolha as opções que correspondem à sua implementação.
8. Aguarde a execução e recupere os resultados em ./situações_de_projeto/situação_exemplo/resultados.

## Sobre o Autor
Gustavo Mello é um estudante de Engenharia Civil da Universidade Federal de Sergipe apaixonado por computação científica
e pelo emprego de aprendizado de máquina na solução de problemas de ciência, engenharia e negócios.
