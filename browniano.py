import numpy as np
import random
from adveccao import Assimilacao
from dominio import Dominio
from rich.traceback import install
install()

class MovimentoBrowniano(Assimilacao):

    """
    =========================================
    Construindo o movimento browniano 
    =========================================
    
    OBJETIVO:
    ---------------
    Construir um caminho aleatório para validação do resultado teórico

    BIBLIOGRAFIA BÁSICA
    --------------

    Higham, D. J. (2001). An algorithmic introduction to numerical simulation of stochastic differential equations. SIAM review, 43(3), 525-546.
    Evans, L. C. (2012). An introduction to stochastic differential equations (Vol. 82). American Mathematical Soc..

    ##### Definição:
    O movimento movimento browniano escalar padrão $W(t)$ em $[0,T]$ é uma variável aleatória que depende continuamente de $t\\in [0.T]$
    e satisfaz as seguintes condições 
    1. W(0) = 1 com probabilidade 1
    2. Para 0<= s<t<= T o incremento W(s)-W(t) ~ N(0,s-t)
    3. Para 0<=s<t<u<v<= T os incrementos W(t)-W(s) e W(v)-W(u) são independentes

    """

    def __init__(self,
                 dom: Dominio,
                 M: int,
                 T: int = 1,
                ):
        
        self.dom = dom
        self.T = T # tempo de execução do movimento Browniano
        self.M = M
        self.dt = self.T/self.M

        
    
    def incrementos(self, t:float = 0):    
        return np.sqrt(self.T-t)*np.sqrt(self.T/self.M)*np.random.randn(self.M)
    
    def bronwniano(self):
        np.random.seed(100)
        
        W = np.zeros(self.M)
        W[0] = self.incrementos()[0]
        for i in range(1, self.M-1):
            print(i)
            W[i] = W[i-1]+self.incrementos()[i]  
        

              
        return W
    
    def matriz(self):
        matrizb = np.zeros((len(dom.x), len(self.n_amostras)))
        for j, tj in enumerate(self.n_amostras):
            matrizb[:, j] = np.random.normal(0,np.sqrt(dom.T - tj), len(dom.x))
    
    " O movimento browniano ocorre no tempo t e não no espaço. CUIDADO!"



if __name__ == "__main__":
    import dominio
    import construtor_de_graficos as cdg
    from adveccao import Assimilacao

    import numpy as np
    import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D
    
    dom = dominio.Dominio()
    #ass = Assimilacao()
    mov = MovimentoBrowniano(dom, M = 5)

    print(f'Valor dos brownianos {mov.bronwniano()}')
    print(f'Valores dos incrementos {mov.incrementos()}')
    

