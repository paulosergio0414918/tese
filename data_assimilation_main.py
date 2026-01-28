import construtor_de_graficos as cdg
import condicoes_iniciais as ci
import metodos_numericos as mn

class DadosDaAssimilacao:
    """ 
    Todos os dados necessários para assimilação estarão aqui
    """

    def __init__(self, 
                 n_testes: int = 32,
                 n_amostras: int = 2,
                 ):
        self.n_testes = n_testes
        self.n_amostras = n_amostras
        



#criando objeto eixo
eixo = mn.SolucaoAdveccao()

#criando o eixo x
x = eixo.eixo_x()

#criando o objeto condição inicial
condicao = ci.Funcoes2d()

#criando o eixo y
y = condicao.condicao_inicial(x)

#criando o objeto grafico
grafico = cdg.Grafico2d(x,y)

#plotando o gráfico
grafico.plot2d()












## Dados da Assimilação segundo o paper 

# criando o objeto condição inicial

#condicao_inicial = ci.Funcoes2d()

# criando o objeto que gera os gráficos