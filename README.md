# Data Assimilations With Python


## INTRODUÇÃO
Aqui temos o objetivo de construir um código de assimilação de dados para a equação da adveção na tentativa de reproduzir os resultados já consolidados no texto 1. Posteriormente desejamos entender o comportamento da assimilação de dados caso os dados apresentem ruído ainda considerando a equação da adveção. e Finalmente desejamos ampliar o resultado do texto 1, desta vez considerando dados ruidosos resultado que não foi estudado no presente texto.

Nesta versão do código, já implementamos a assimilação de dados para advecção e validamos o nosso resultado. Provamos que a norma da diferença entre a condição inicial $\phi^{(t)}(x)$ e a melhor condição inicial obtida pelo processo de assimilação de dados 4dvar $\phi^{(b)}(x)$ é diretamente proporcinal à soma da norma dos ruidos na amostras $|E|$ e inversamente proporcional ao número de amostras $N$.
```math
|\phi^{(t)}(x) - \phi^{(b)}(x)| \approx \frac{|E|}{N}
```

Para águas rasas linearizado implementamos quatro métodos de solução numéricos. Dois de diferenças finitas, FTCS e leapfrog e dois de discretização temporal para o método de volumes finitos SSPRK22 e SSPRK33 que foram utilizados em no texto 1. que podem ser encontrado em 3.. A execução de todos os código apresentam instabilidades numéricas considerando diversas opções para o número de Courant.

Instabilidades estas a serem corrigidas na próxima atualização.

## INSTRUÇÕES

Afim de facilitar a correção de possíveis bugs, cada aspecto do código foi dividido em arquivos separados.
Desta forma:
* O arquivo construto_de_graficos.py é uma forma mais rápida de construir gráficos.
* Todas as funções usadas estão no arquivo condicoes_iniciais.py
* O arquivo dominio.py constoe o dominio espacial e temporal a ser discretizado, constuido a malha a ser usado em todos os teste.
* Os métodos de discretização da equação de adveção quanto sua validação com a teoria e a assimilação está no arquivo adveccao.py. Exemplos podem ser encontrados ao final do arquivo.
* Os códigos utilizados para águas rasas linearizado estão presente no arquivo aguas_rasas.py.


Em todos os arquivos citados existem exemplos de execução em if __name__ == "__main__" no final de cada arquivo.


## CODE ISSUES ADVECÇÃO
- [x] Conseguir apresentar gráficos de forma eficiente
- [x] Calcular as condições iniciais
- [x] Coletar os dados através das condições inicias
- [x] Obter a assimilação de dados de forma numérica como dado em 2.
- [x] Obter a assimilação de dados de forma analítica como proposto em minha tese

## CODE ISSUES ÁGUAS RASAS LINEARIZADA
- [x] Conseguir apresentar gráficos de forma eficiente
- [x] Calcular as condições iniciais
- [] Coletar os dados através das condições inicias
- [] Obter a assimilação de dados de forma numérica como dado em 2.
- [] Obter a assimilação de dados de forma analítica como proposto em minha tese

## CODE ISSUES ÁGUAS RASAS PLANO F
- [] Conseguir apresentar gráficos de forma eficiente
- [] Calcular as condições iniciais
- [] Coletar os dados através das condições inicias
- [] Obter a assimilação de dados de forma numérica como dado em 2.
- [] Obter a assimilação de dados de forma analítica como proposto em minha tese


## BIBLIOGRAFIA BÁSICA
1. Kevlahan, N. R., Khan, R., & Protas, B. (2019). On the convergence of data assimilation for the one-dimensional shallow water equations with sparse observations. Advances in Computational Mathematics, 45(5), 3195-3216.

2. Kalnay, E. (2003). Atmospheric modeling, data assimilation and predictability. Cambridge university press. 

3. Spiteri, R., Ruuth, S.: A new class of optimal high-order strong-stability-preserving time discretization methods. SIAM J. Numer. Anal. 40, 469–491 (2002)