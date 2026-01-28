import numpy as np
from typing import Union

class Funcoes2d:
    """
    Classe que fornece todas as condições iniciais estudadas durante a pesquisa.
    """

    def __init__(self):
        pass
    
    def __str__(self):
        return "Classe destinada a construção de condições iniciais para o " \
        "processo de assimilação de dados. As funções aqui presentes são condisentes" \
        "com a bibliografia utlizada na tese. "

    def condicao_paper(self,
                         x: Union[float, tuple]
                         ) -> tuple:
        
        return (1/20)*np.exp(-100*x**2)
    
    def condicao_inicial2(self,
                          x: Union[float, tuple]
                          )-> tuple:
        return (1/20)*np.exp(-1_000*x**2)
    
    def condicao_caixa(self, 
                       x: Union[float, tuple]
                       )-> tuple:
        if isinstance(x, (np.ndarray, list, tuple)):
            x = np.asarray(x)
            return 0.05*((x >= -1).astype(float) + (x <= 1).astype(float) - 1)
        else:
            return 0.05*(float(x >= -1) + float(x <= 1) - 1)
    


if __name__ == "__main__":
    import construtor_de_graficos as cg

    #construindo o eixo x conforme o paper
    x = np.linspace(-1.5, 1.5, 1_000)

    #construindo o objeto condicao inicial
    cod_inicial = Funcoes2d()

    #construindo o eixo y
    y = cod_inicial.condicao_paper(x)
    y2 = cod_inicial.condicao_caixa(x)

    #construindo o grafico da função
    grafico_duplo = cg.Grafico2d(y,y2)
    grafico_duplo.duplo_plot()