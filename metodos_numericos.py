import numpy as np
from rich.traceback import install
import random
install()

import construtor_de_graficos as cdg
import condicoes_iniciais as ci


class SolucaoAdveccao:
    """
    Aqui faremos todos os códigos responsáveis pela discretização das equações
    E coleta de amostras que serão usadas na assimilação dicretizada
    """
    

    def __init__(self,
                 condicao: str = "condicao_paper",
                 iteracoes: int = 100, # numero de iterações do metodo de solucao
                 a: int = 1, #velocidade de adveccao
                 L: float = 4, #Largura do domínio espacial periódico [-L, L]
                 N: int = 4096, # numero de passos no espaço
                 T: float = 2, # Domínio temporal [0, T]
                 M: int = 1024 # número de passos no tempo  
                ):
        
        self.condicao = condicao
        self.iteracoes = iteracoes
        self.a = a 
        self.L = L
        self.N = N
        self.T = T
        self.M = M

    def CFL(self):
        """Calcula o número CFL (Condição de Courant-Friedrichs-Lewy)"""
        return (2*self.a*self.L*self.M)/(self.N*self.T)


    def validar_cfl(self):
        pass
        #try: 
            #cfl = (2*self.a*self.L*self.M)/(self.N*self.T)
            #Ret
        #except cfl != 1:
            #self.L = 


    def eixo_x(self):
        """Define o eixo x"""
        return np.linspace(-self.L, self.L, self.N)
    
    def u_zero(self):
        """Define a condição inicial no eixo u"""
        condicao = ci.Funcoes2d()

        if self.condicao == "condicao_caixa":
            return condicao.condicao_caixa(self.eixo_x())
        
        else:
            return condicao.condicao_paper(self.eixo_x())

    
    def Lax_Friedrichs(self,
                       u: tuple ):
        """Resolução da equação da advecção """
        cfl = self.CFL()
        return 0.5*((1 + cfl)*np.roll(u,1) + (1 - cfl)*np.roll(u,-1))
    
    def solucao(self):
        """Calcula a solução da advecção após vários instantes."""
        solucao = self.u_zero()

        for _ in range(self.iteracoes):
            solucao = self.Lax_Friedrichs(solucao)

        return solucao
    
class Amostras(SolucaoAdveccao):
    """ Classe destinada a coletar as amostras para assimilação."""
    def __init__(self,
                 N: int = 4096,
                 T: int = 2,
                 M: int = 1024,
                 n_amostras: int = 2,
                 ):  
        super().__init__(N, T, M)
        self.n_amostras = n_amostras
    
    def matriz_de_amostras(self):
        matriz = np.zeros((self.N ,self.n_amostras))
        passo = int(np.floor(self.M/self.T))

        for i in range(self. n_amostras):
            resposta = SolucaoAdveccao(iteracoes = (i+1)*passo)
            matriz[:, i] = resposta.solucao()

        return matriz
    def matriz_de_amostras_ruido(self):
        matriz_com_ruido = self.matriz_de_amostras()
        for i in range(self.N):
            for j in range(self.n_amostras):
                matriz_com_ruido[i,j] += random.uniform(0, 0.0005)
        return matriz_com_ruido
        




if __name__ == "__main__":

    #criando o objeto amostras
    amostragem = Amostras(n_amostras = 2)

    #criando os vetores
    x = amostragem.matriz_de_amostras_ruido()[:,0]
    y = amostragem.matriz_de_amostras_ruido()[:,1] 

    # construindo o objeto solução
    #resposta = SolucaoAdveccao(iteracoes = 1_000)
    #x = resposta.eixo_x()
    #y = resposta.u_zero()
    #sol = resposta.solucao()
    #print(resposta.CFL())
    #construindo o objeto grafico
    grafico = cdg.Grafico2d(x,y)
    grafico.duplo_plot()
         