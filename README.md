# Data Assimilations With Python


## INTRODUÇÃO
Aqui temos o objetivo de construir um código de assimilação de dados para a equação da adveção na tentativa de reproduzir os resultados já consolidados no texto 1. Posteriormente desejamos entendo o comportamento da assimilação de dados caso os dados apresentem ruído ainda considerando a equação da adveção. e Finalmente desejamos ampliar o resultado do texto 1, desta vez considerando dados ruidosos resultado que não foi estudado no presente texto.

## INSTRUÇÕES

Afim de facilitar a correção de possíveis bugs, cada aspecto do código foi dividido em arquivos separados.
Desta forma:
* Todos os gráficos são constuidos pela classe Grafico2d no arquivo construto_de_graficos.py
* Todas as funções usadas estão no arquivo condicoes_iniciais.py
* O método de discretização se encontra no arquivo metodos_numericos.py juntamento com a função que coleta os dados utilizando a condição inicial dada.

E todos as funcionalidades são concentradas e realizadas no arquivo data_assimilation_main.py
Cujas instruções podem ser obtidas aqui ou pelo código inspect(objeto) no terminal.


## CODE ISSUES
- [x] Conseguir apresentar gráficos de forma eficiente
- [x] Calcular as condições iniciais
- [x] Coletar os dados através das condições inicias
- [ ] Obter a assimilação de dados de forma numérica como dado em 2.
- [ ] Obter a assimilação de dados de forma analítica como proposto em minha tese

## BIBLIOGRAFIA BÁSICA
1. Kevlahan, N. R., Khan, R., & Protas, B. (2019). On the convergence of data assimilation for the one-dimensional shallow water equations with sparse observations. Advances in Computational Mathematics, 45(5), 3195-3216.

2. Kalnay, E. (2003). Atmospheric modeling, data assimilation and predictability. Cambridge university press. 