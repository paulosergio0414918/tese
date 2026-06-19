import numpy as np
from rich.traceback import install
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
install()

import condicoes_iniciais as ci
from dominio import Dominio


class SolucaoAguasRasas:
    """ Solucao analitica e numerica da equação de águas rasas."""

    def __init__(self,
                 dom: Dominio,
                 condicao: str = "condicao_paper"
                 ):
        
        self.dom = dom
        self.condicao = condicao
        self.variacao_energia = 0
    
    def eta_zero(self, 
                 x: np.ndarray = None) -> np.ndarray:
        """Define a condição inicial para variável η"""
        condicao = ci.Funcoes2d()
        if x is None:
            x = self.dom.x
        if self.condicao == "condicao_caixa":
            return condicao.condicao_caixa(x)
        
        else:
            return condicao.condicao_paper(x)

    def u_zero(self,
               x: np.ndarray = None) -> np.ndarray:
        """Define a condição inicial para variável u"""
        if x is None:
            x = self.dom.N
            return np.zeros(x)
        
        elif isinstance(x,np.ndarray):
            return np.zeros(len(x))

    def solucao_analitica_eta(self,
                          tempo: int = None
                          ) -> np.ndarray:
        """Fornece a solução analítica para η"""
        if tempo is None:
            tempo = self.dom.M   

        return 0.5*(self.eta_zero(self.dom.x - (tempo+1)*self.dom.dt) \
            + self.eta_zero(self.dom.x + (tempo+1)*self.dom.dt))
    
    def solucao_analitica_u(self,
                          tempo: int = None
                          ) -> np.ndarray:
        """Fornece a solução analítica para η"""
        if tempo is None:
            iteracao = self.dom.M   

        return 0.5*(self.eta_zero(self.dom.x - (tempo+1)*self.dom.dt) \
            - self.eta_zero(self.dom.x + (tempo+1)*self.dom.dt))
    
    def calculo_cfl(self,
                    n: int = None,
                    m: int = None):
        if n is None:
            n = self.dom.N
        
        if m is None:
            m = self.dom.M

        lam = ((self.dom.T-self.dom.t0)*n)/(2*self.dom.L*m)
        if lam >1 or lam<0:
            return "Instável"
        else:
            return lam
     
    def ftcs(self,
             eta: np.ndarray = None,
             u: np.ndarray = None):
        """Avança uma unidade de tempo usando diferenças centradas no 
         espaço e diferença avançada no tempo. """
        if eta is None:
            eta = self.eta_zero()
        if u is None:
            u = self.u_zero()

        cfl = self.calculo_cfl()
        v = np.roll(u,-1)-np.roll(u,1)
        w = np.roll(eta,-1)-np.roll(eta,1)

    
        return {
                'eta_final' : eta - v*cfl*0.5,
                'u_final': u - w*cfl*0.5
            }

    def leapfrog(self,
                eta: np.ndarray = None,
                u: np.ndarray = None,
                t: int = None):
        """Avança uma unidade de tempo usando diferenças centradas no 
         espaço e diferença centradas no tempo. """
        
        if eta is None:
            eta = self.eta_zero()
        if u is None:
            u = self.u_zero()
        if t is None:
            t = self.dom.M
        #inicialializando o método que usa dois passos desconectados no tempo
        matriz_eta = np.zeros((self.dom.N, 2))
        matriz_u = np.zeros((self.dom.N, 2))
        #print(f"tamanho da matriz eta{matriz_eta.shape}")
        #print(f"tamanho da matriz u {matriz_u.shape}")
        #primeira coluna com as condições inciais
        matriz_eta[:,0] = eta
        matriz_u[:,0] = u
        #segunda coluna com um passo do FTCS
        solucao_n1 = self.ftcs(matriz_eta[:,0],matriz_u[:,0])
        matriz_eta[:,1] = solucao_n1['eta_final']
        matriz_u[:,1] = solucao_n1['u_final']
        #terceira coluna atualizada com o metodo leapfrog
        cfl = self.calculo_cfl()
        #v = np.roll(matriz_u[:,1],-1)-np.roll(matriz_u[:,1],1)
        #w = np.roll(matriz_eta[:,1],-1)-np.roll(matriz_eta[:,1],1)
        #matriz_eta[:,2] = matriz_eta[:,0]-cfl*v
        #matriz_u[:,2] = matriz_u[:,0]-cfl*w
        for i in range(t):
            # executando o loop espacial
            v = np.roll(matriz_u[:,1],-1)-np.roll(matriz_u[:,1],1)
            w = np.roll(matriz_eta[:,1],-1)-np.roll(matriz_eta[:,1],1)
            #"inserindo a coluna")
            matriz_eta = np.column_stack((matriz_eta, matriz_eta[:,0]-cfl*v))
            matriz_u = np.column_stack((matriz_u,matriz_u[:,0]-cfl*w))
            #"excluindo a coluna")
            matriz_eta = np.delete(matriz_eta, 0, axis=1)
            matriz_u = np.delete(matriz_u, 0, axis=1)

        return {
                'eta_final' : matriz_eta[:,1],
                'u_final': matriz_u[:,1]
            }

    def forcante(self,
                u: np.ndarray = None,
                eta: np.ndarray = None
                ): # forçante do método de volumes finitos

        if eta is None:
            eta = self.eta_zero()
             
        if u is None:
            u = self.u_zero()
            

        u1 = np.roll(u, -1) #retrocede um índice no vetor u
        eta1 = np.roll(eta, -1) #retrocede um índice no vetor eta

        deta_dt = (1/self.dom.dx)*(u-u1)
        du_dt = (1/self.dom.dx)*(eta-eta1)
        return {
                'deta_dt' : deta_dt,
                'du_dt': du_dt
            }   
    
    def ssprk22(self,
                cond_eta: np.ndarray = None,
                cond_u: np.ndarray = None,
                t: int = None):
        """Avança uma unidade de tempo usando Runge-Kutta 22 """
        #################
        ####corrigir o valores padrões
        #################

        propagacao = self.forcante(self.eta_zero(self.dom.x_centro), self.u_zero(self.dom.x_borda))
        if cond_eta is None:
            cond_eta = propagacao['deta_dt']
        if cond_u is None:
            cond_u = propagacao['du_dt']
        if t is None:
            t = self.dom.M
        for i in range(t):

            #primeiro estágio
            propagacao_1 = self.forcante(cond_eta, cond_u)
            eta_1 = cond_eta + self.dom.dt*propagacao_1["deta_dt"]
            u_1 = cond_u + self.dom.dt*propagacao_1["du_dt"]

            #segundo estágio
            propagacao_2 = self.forcante(eta_1, u_1)
            eta_2 = 0.5*cond_eta + 0.5*eta_1 + self.dom.dt*propagacao_2["deta_dt"]
            u_2 = 0.5*cond_u + 0.5*u_1 + self.dom.dt*propagacao_2["du_dt"]        

            # atualização para reiniciar o loop temporal
            cond_eta = eta_2
            cond_u = u_2


        return {
                'eta_final' : eta_2,
                'u_final': u_2
            } 
    
    def ssprk33(self,
            cond_eta: np.ndarray = None,
            cond_u: np.ndarray = None,
            t: int = None):
        """Avança uma unidade de tempo usando Runge-Kutta 33 """
      
        propagacao = self.forcante(self.eta_zero(), self.u_zero())
        if cond_eta is None:
            cond_eta = propagacao['deta_dt']
        if cond_u is None:
            cond_u = propagacao['du_dt']
        if t is None:
            t = self.dom.M
        for i in range(t):

            #primeiro estágio
            propagacao_1 = self.forcante(cond_eta, cond_u)
            eta_1 = cond_eta + self.dom.dt*propagacao_1["deta_dt"]
            u_1 = cond_u + self.dom.dt*propagacao_1["du_dt"]
      
            
            #segundo estágio
            propagacao_2 = self.forcante(eta_1, u_1)
            eta_2 = 0.75*cond_eta + 0.25*eta_1 + 0.25*self.dom.dt*propagacao_2["deta_dt"]
            u_2 = 0.75*cond_u + 0.25*u_1 + 0.25*self.dom.dt*propagacao_2["du_dt"]        
    
            
            #terceiro estágio
            propagacao_3 = self.forcante(eta_2, u_2)
            eta_3 = (1/3)*cond_eta + (2/3)*eta_2 + (2/3)*self.dom.dt*propagacao_3["deta_dt"]
            u_3 = (1/3)*cond_u + (2/3)*u_2 + (2/3)*self.dom.dt*propagacao_3["du_dt"]        
   
            
            # atualização para reiniciar o loop temporal
            cond_eta = eta_3
            cond_u = u_3


        return {
            'eta_final' : eta_3,
            'u_final': u_3
        } 

    def solucao_numerica(self,
                         solucao_eta:  np.ndarray = None, # condição incial para eta
                         solucao_u: np.ndarray = None, #condição inicial para u
                         tempo: int = None, # tempo de execução do método
                         modo: str = "leapfrog" # modelo de execução
                         ):
        """Calcula a solução de águas rasas após vários instantes."""

        if tempo is None:
            tempo = self.dom.M
            
        if solucao_eta is None:
            solucao_eta = self.eta_zero()
    
        if solucao_u is None:
            solucao_u = self.u_zero()
            
        if modo == "ftcs":
            propagacao = self.ftcs(eta = solucao_eta, u= solucao_u)
            for _ in range(tempo):
                eta_final = propagacao['eta_final']
                u_final = propagacao['u_final']
                propagacao = self.ftcs(eta = eta_final, u= u_final)

        elif modo == "leapfrog":
            
            propagacao = self.leapfrog(solucao_eta,solucao_u, t = tempo)
            eta_final = propagacao['eta_final']
            u_final = propagacao['u_final']
        
        elif modo == "ssprk22":
        
            propagacao = self.ssprk22(solucao_eta,solucao_u, t = tempo)
            eta_final = propagacao['eta_final']
            u_final = propagacao['u_final']            

        elif modo == "ssprk33":
        
            propagacao = self.ssprk33(solucao_eta,solucao_u, t = tempo)
            eta_final = propagacao['eta_final']
            u_final = propagacao['u_final'] 
        
        else:
            print("Modo não definido")

        
        return {
                'eta' : eta_final,
                'u'   : u_final 
            }


class Validacao(SolucaoAguasRasas):
    """validação do método numérico"""
  
    def __init__(self,
                dom: Dominio,
                testes: int = 6
                ):
        self.testes = testes
        self.delta_E = 0
        self.energia_total = 0
        self.dom = dom
    
    def valores_cfl(self):
        from rich.table import Table
        from rich import print
        passo = 1024
        tab = Table(title = " Número de Courant e Energia do Sistema.")
        tab.add_column(f" ", justify = "center")
        tab.add_column(f"N = {int(passo/(2**2))}", justify = "center")
        tab.add_column(f"N = {int(passo/(2**1))}", justify = "center")
        tab.add_column(f"N = {passo} ", justify = "center")
        tab.add_column(f"N = {passo*(2**1)}", justify = "center")
        tab.add_column(f"N = {passo*(2**2)}", justify = "center")

        for j in range(10):
            tab.add_row(f"M = {2**(j+5)}",
                        f"λ = {self.calculo_cfl(passo/(2**2),2**(j+5))}",
                        f"λ = {self.calculo_cfl(passo/(2**1), 2**(j+5))}",
                        f"λ = {self.calculo_cfl(passo, 2**(j+5))}",
                        f"λ = {self.calculo_cfl(passo*(2**1), 2**(j+5))}", 
                        f"λ = {self.calculo_cfl(passo*(2**2), 2**(j+5))}"
                        )
        print(tab)
    
    def calculo_energia(self,
                        solucao_eta: np.ndarray = None,
                        solucao_u:np.ndarray = None,
                        dx: float = None
                        ) -> float:
        #aqui eu quero calcular a energia dada uma solucao
        return  np.sum(solucao_eta**2 + solucao_u**2) * dx

    def variacao_de_energia(self,
                      n: int = None,
                      m: int = None,
                      t: int = None,
                      modo: str = "ftcs"
                      ) -> float:
        # aqui eu quero calcular a variação total da energia do método numérico
        if n is None:
            n = self.dom.N
        if m is None:
            m = self.dom.M
        if t is None:
            t = self.dom.M
        
        #gerar os objetos para a solucao
        domi = dominio.Dominio(N = n, M = m)
        sol = SolucaoAguasRasas(domi)
        eta = sol.eta_zero(domi.x)
        u = sol.u_zero(domi.N)

        #solucao no inicio do intervalo
        propagacao1 = sol.solucao_numerica(tempo= 1, solucao_eta=eta, solucao_u=u, modo = modo)
        solucao_eta1 = propagacao1['eta']
        solucao_u1 = propagacao1['u']
        energia_inicial =  np.sum(solucao_eta1**2 + solucao_u1**2) * domi.dx

        #solucao no fim do intervalo
        propagacao2 = sol.solucao_numerica(tempo= t, solucao_eta=eta, solucao_u=u, modo = modo)
        solucao_eta2 = propagacao2['eta']
        solucao_u2 = propagacao2['u']
        energia_final =  np.sum(solucao_eta2**2 + solucao_u2**2) * domi.dx
        
        #atualiza o valor da energia
        self.delta_E = np.abs(energia_final-energia_inicial) / energia_inicial
        print(f"Erro relativo final de conservação de Energia tomando o método {modo}.")
        return   self.delta_E
        
    def vetor_energia(self,
                    n: int = None,
                    m: int = None,
                    t: int = None,
                    modo: str = "ftcs") -> float:

        """ cria um vetor armazenando toda a energia no intervalo temporal"""
        if n is None:
            n = self.dom.N
        if m is None:
            m = self.dom.M
        if t is None:
            t = self.dom.M        
        vetor = []
        domi = dominio.Dominio(N = n, M = m)
        sol = SolucaoAguasRasas(domi)
        eta = sol.eta_zero(domi.x)
        u = sol.u_zero(domi.N)
        
        sol_atualizada = sol.solucao_numerica(solucao_eta = eta, solucao_u = u,tempo = 10, modo = modo )
        eta = sol_atualizada['eta']
        u = sol_atualizada['u']
        E1 = self.calculo_energia(solucao_eta=eta,solucao_u=u,dx=domi.dx)
        #vetor.append(float(E1))

        for _ in tqdm(range(t), desc = "processando"):
            sol_atualizada = sol.solucao_numerica(solucao_eta = eta, solucao_u = u,tempo = 10, modo = modo )
            eta = sol_atualizada['eta']
            u = sol_atualizada['u']
            E2 = self.calculo_energia(solucao_eta=eta,solucao_u=u,dx=domi.dx)
            vetor.append(float(np.abs(E2-E1)/E1))
            E1=E2

        return vetor

    def evolucao_da_energia(self,
                    n: int = None,
                    m: int = None,
                    tempo: int = None,
                    modo: str = "ftcs"
                    ):

        """ cria um vetor armazenando toda a energia no intervalo temporal"""
        if n is None:
            n = self.dom.N
        if m is None:
            m = self.dom.M
        if tempo is None:
            tempo = self.dom.M  
        cfl = val.calculo_cfl(n, m)
        x = [i+10 for i in range(tempo)]
        y = self.vetor_energia(n, m , t = tempo, modo = modo)
        plt.scatter(x, y)
        plt.title(f"Evolução da energia para {modo}.")
        plt.yscale('log')
        plt.show()
     

class Assimilacao(SolucaoAguasRasas):

    def __init__(self,
                 dom: Dominio, # um domínio criado pela classe Dominio
                 c: float = 1, #velocidade de advecção
                 n_amostras: int = 2,
                 condicao: str = "condicao_paper",
                 ruido: bool = False,
                 modo: str = "analitico"
                 ):  
        self.n_amostras = n_amostras
        self.dom = dom
        self.c = c
        self.condicao = condicao
        self.passos = [int((dom.M*(dom.T-(dom.T/self.n_amostras)*i))/2) for i in range(self.n_amostras)]
        self.ruido = ruido
        self.sol = SolucaoAguasRasas(self.dom)
        self.modo = modo
        self._matriz_com_amostras = None
        self._matriz_com_amostras_ruido = None
        self.vetor_custo = []
        self.vetor_ruido = [random.uniform(0, 0.005) for i in range(self.dom.N)]
        self.E = np.linalg.norm(self.vetor_ruido)/self.n_amostras
        self.tj = [((dom.M*(dom.T-(dom.T/self.n_amostras)*i))/2)*dom.dt for i in range(self.n_amostras)]

        self.matriz_de_amostras_ruido()

    def matriz_de_amostras(self):
        #if self._matriz_com_amostras is None:
        """Gera uma matriz contendo as amostras sem perturbação"""
        matriz = np.zeros((self.dom.N ,self.n_amostras))
        solu = SolucaoAguasRasas(self.dom, self.condicao)
        
        for i, passo in enumerate(self.passos):
            
            matriz[:, i] = solu.solucao_analitica_eta(iteracao=passo)
            
        self._matriz_com_amostras = matriz
        return matriz
        
    def matriz_de_amostras_ruido(self):
        """Gera uma matriz contendo as amostras com perturbação """
        #if self._matriz_com_amostras_ruido is None:
        if self._matriz_com_amostras is  None:
            self.matriz_de_amostras()     
        matriz_com_ruido = self._matriz_com_amostras.copy()
        
        for j in range(self.n_amostras):
            matriz_com_ruido[:,j] += self.vetor_ruido
        self._matriz_com_amostras_ruido = matriz_com_ruido
        return matriz_com_ruido    





if __name__ == "__main__":
    import dominio
    import matplotlib.pyplot as plt
    import construtor_de_graficos as cdg
    import numpy as np

    dom = Dominio(M=4096, N=1024)
    sol = SolucaoAguasRasas(dom)
    val = Validacao(dom)
    op = 2
    it = 256


    if op == 4: # teste da evolução da energia do sistema
        val.evolucao_da_energia(modo = "leapfrog")

    elif op == 3:#teste do calculo da energia do sistema
        print(f'variação de energia = {val.variacao_de_energia()}')
        print(f'cfl = {sol.calculo_cfl()}')
   
    elif op == 2: # teste da solução numerica para η
        for i in range(it):

            solucao = sol.solucao_numerica(modo= "ssprk22", tempo = i+1)
            solucao2 = sol.solucao_numerica(modo= "ssprk33" ,tempo = i+1)
            z = sol.solucao_analitica_eta(tempo = i)
            y = solucao['eta']
            e_22 = np.linalg.norm(z-y)
            y2 = solucao2['eta']
            e_33 = np.linalg.norm(z-y2)
            
            if (e_22 or e_33) > 1:
                print(f"erro leap frog { e_22}")
                print(f"erro leap ftcs { e_33}")
                print("erro muito grande.")
                break
            plt.clf()
            plt.xlim(-dom.L, dom.L) # x limit
            plt.plot(dom.x, y, label = 'Solução numérica ssprk22' )
            plt.plot(dom.x, y2, label = 'Solução numérica ssprk33' )
            plt.plot(dom.x, z, label = 'Solução Analítica')
            plt.title(f'Execução {i+1} de {it}.')
            plt.legend()

            #plt.show(block = False)
            plt.pause(0.01)

        plt.show()               

    elif op == 1: # teste da cfl
        print(sol.calculo_cfl())

    elif op == 0: # teste da solução analítica para η
        y = sol.solucao_analitica_eta()

        graf = cdg.Grafico2d(dom.x,y)
        graf.plot2d()

