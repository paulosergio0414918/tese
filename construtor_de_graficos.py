import matplotlib.pyplot as plt
import numpy as np

class Grafico2d:
    """
    -> Construtor de gráficos de funções de duas dimenções
    x: tupla com valores do eixo x
    y: tupla com valores do eixo y
    title: Nome do gráfico a ser construido 
    """
    def __init__(self, 
                 x: tuple, 
                 y: tuple,
                 inicio: float = 4, 
                 title: str = " Grafico da função",
                 passos: int = 4096 
                 ):
       
       self.title = title
       self.x = x
       self.y = y
       self.passos = passos
       self.inicio = inicio
       plt.ylim(-0.025, 0.06) #limites do eixo y
       plt.xlim(-0.5, 4) #limites do eixo x
       plt.title(self.title) # título do Gráfico
       
       
    
    def __str__(self):
        return 'Produz gráficos de funções de duas dimensões usando o método plot2d(x,y). ' \
        'Ou produz um gráfico com duas funções usando o método plot_duplo(funcao1, funcao2).'

    def plot2d(self):
        plt.plot(self.x, self.y)
        plt.show()
        return self
    
    def duplo_plot(self):
        x = np.linspace(-self.inicio, self.inicio, self.passos)
        plt.plot(x, self.x)
        plt.plot(x, self.y)
        plt.show()
        return self
    
if __name__ == "__main__":
    def funcao(x):
        return x/100
    
    x = np.linspace(-1, 1, 1_000)
    y = funcao(x)
    grafico = Grafico2d(x,y)
    grafico.plot2d()