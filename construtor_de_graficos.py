import matplotlib.pyplot as plt
import numpy as np


class Grafico2d:
    """
    MANUAL DE USO - Grafico2d
    =======================================
    Esta classe constrói gráficos em duas dimensões

    INICIALIZAÇÃO
    ---------------------------------------
    graf = Grafico2d(x, y1, y2, inicio, title)

    PARÂMETROS
    ---------------------------------------
    x (tuple): vetor para o eixo horizontal
    y1 (tuple): vetor para o eixo vertical
    y2 (tuple, opcinal): vetor para o eixo vertical
    inicio (float, opcional): valor inicial do eixo horizontal
    title (str, opcional): nome para o gráfico

    MÉTODOS PRINCIPAIS
    --------------------------------------
    1. plot2d(x,y):
        retorna o gráfico para a função dada em y

    2. plot2d(x,y1,y2):
        retorna o grafico contendo as duas funções y1 e y2

    ATENÇÃO:
    --------------------------------------
    Cuidado com o tamanho dos eixos, pois devem ser compatíveis
    recomendamos importar o a classe Domínio, pois nela está
    concentrada todos os dados utilizado no artigo na construção
    deste código.

    EXEMPLO:
    --------------------------------------
    import dominio
    import condicoes_iniciais as cond
    #criando o domínio do paper 
    dom = dominio.Dominio()
    u0 =  cond.Funcoes2d()
    y = u0.condicao_paper(dom.x)
    y2 = u0.condicao_caixa(dom.x)
    #plotando os graficos
    grafico = Grafico2d(dom.x, y, y2, inicio = dom.x[0] , title= "Condição incial do Paper")
    grafico.plot2d()

    """
    def __init__(self, 
                 x: tuple, 
                 y1: tuple,
                 y2: tuple = None,
                 inicio: float = 4, 
                 title: str = " Grafico da função",
                 y1_name: str = "Grafico 1",
                 y2_name: str = "Grafico 2"
                 ):
       
       self.title = title
       self.y2 = y2
       self.x = x
       self.y1 = y1
       self.inicio = inicio
       self.y1_name = y1_name
       self.y2_name = y2_name
       plt.xlim(-self.inicio, self.inicio) #limites do eixo x
       plt.title(self.title) # título do Gráfico
        
       
    
    def __str__(self):
        return 'Produz gráficos de funções de duas dimensões usando o método plot2d(x,y). ' \
        'Ou produz um gráfico com duas funções usando o método plot_duplo(funcao1, funcao2).'

    def plot2d(self):
        if self.y2 is not None:
            plt.plot(self.x, self.y2, label = self.y2_name)
            plt.plot(self.x, self.y1, label = self.y1_name)
            plt.legend()
            plt.show()

        else:
            plt.plot(self.x, self.y1, label = self.y1_name)
            plt.legend()
            plt.show()

        
        return self
    
if __name__ == "__main__":
    import dominio
    import condicoes_iniciais as cond
    #criando o domínio do paper 
    dom = dominio.Dominio()
    u0 =  cond.Funcoes2d()
    y = u0.condicao_paper(dom.x)
    y2 = u0.condicao_caixa(dom.x)
    grafico = Grafico2d(dom.x, y, y2, inicio = dom.x[0], y1_name="funcao 1" , title= "Condição incial do Paper")
    grafico.plot2d()