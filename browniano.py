import numpy as np
import random
from adveccao import Assimilacao
from dominio import Dominio
from condicoes_iniciais import Funcoes2d
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
                 T: int = 2,
                 n_amostras: int = 2 
                ):
        
        self.dom = dom
        self.T = T # tempo de execução do movimento Browniano
        self.M = M
        self.n_amostras = n_amostras
        self.dt = self.T/self.M
        self.tj = [((dom.M*(dom.T-(dom.T/self.n_amostras)*i))/2)*self.dt for i in range(self.n_amostras)]


   
    def bronwniano(self,
                   t: float = 0,
                   seed: int = 100
                   ):
        
        np.random.seed(seed) 
        incrementos = np.sqrt(self.T-t)*np.sqrt(self.T/self.M)*np.random.randn(self.M)        
        W = np.zeros(self.M)
        W[0] = incrementos[0]
        for i in range(1, self.M):
            W[i] = W[i-1] + incrementos[i]  

        return W
    

    def matriz_b(self):
        matrizb = np.zeros((self.M, self.n_amostras))
        for j in range(self.n_amostras):
            matrizb[:, j] = self.bronwniano(t = self.tj[j])

        return matrizb
    



if __name__ == "__main__":
    import dominio
    import construtor_de_graficos as cdg
    from adveccao import Assimilacao

    import numpy as np
    import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D
    
    amos = 10
    dom = dominio.Dominio()
    ass = Assimilacao(dom = dom)
    mov = MovimentoBrowniano(dom, M = dom.M, n_amostras=amos)

      
    matriz = mov.matriz()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(amos):
        ax.plot(np.full_like(dom.t, i+1), dom.t, matriz[:, i], linewidth=2)

    ax.set_xlabel('Amostra')
    ax.set_ylabel('t')
    ax.set_zlabel('W')
    ax.set_xticks([i for i in range(amos)])
    ax.view_init(25, -60)
    plt.show()