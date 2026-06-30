#aguas rasas linear
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
    
    def calculo_cfl(self):

        cfl = self.dom.dt / self.dom.dx
        if cfl >1 or cfl<0:
            print(f"Instável, cfl = {cfl}.")
        
        return cfl
     
    def godunov_euler(
            self,
            eta: np.ndarray = None,
            u: np.ndarray = None
            ) -> np.ndarray:
        
        eta_bar = np.zeros(self.dom.N)
        u_bar = np.zeros(self.dom.N)
        
        for j in range(1,self.dom.N-1):
            eta_bar[j] = eta[j]+(0.5*self.calculo_cfl())*(eta[j+1]-2*eta[j]+eta[j-1]+u[j-1]-u[j+1])
            u_bar[j] = u[j]+(0.5*self.calculo_cfl())*(u[j+1]-2*u[j]+u[j-1]+eta[j-1]-eta[j+1])
        
        #inserindo a condição de fronteira periódica
        eta_bar[0] = eta[0]+(0.5*self.calculo_cfl())*(eta[1]-2*eta[0]+eta[-1]+u[-1]-u[1])
        eta_bar[-1] = eta[-1]+(0.5*self.calculo_cfl())*(eta[0]-2*eta[-1]+eta[-2]+u[-2]-u[0])
        u_bar[0] = u[0]+(0.5*self.calculo_cfl())*(u[1]-2*u[0]+u[-1]+eta[-1]-eta[1])
        u_bar[-1] = u[-1]+(0.5*self.calculo_cfl())*(u[0]-2*u[-1]+u[-2]+eta[-2]-eta[0])
        
        return{
            'eta_final': eta_bar,
            'u_final': u_bar
        }
    
    def muscl_ssprk(self,
                cond_eta: np.ndarray = None,
                cond_u: np.ndarray = None,
                modo: str = "muscl_ssprk33", # modelo de execução

                t: int = None):
        """Avança uma unidade de tempo usando SSPRK22 """
        #construindo as inclinações delta_j

        def delta_minmod(q):# van_leer
            N = len(q)
            d = np.zeros(N)
            # diferenças com periodicidade
            dl = q - np.roll(q, 1)          # q[j] - q[j-1]
            dr = np.roll(q, -1) - q         # q[j+1] - q[j]
            mask = (dl * dr) > 0 #retorna um vetor booleano
            d[mask] = np.where(np.abs(dl) < np.abs(dr), dl, dr)[mask] # seleciona o menor elemento dos vetores dr e dl
            return d

        def delta_van_leer(q):# van_leer
            N = len(q)
            d = np.zeros(N)
            # diferenças com periodicidade
            dl = q - np.roll(q, 1)          # q[j] - q[j-1]
            dr = np.roll(q, -1) - q         # q[j+1] - q[j]
            for i in range (N):
                if dl[i]*dr[i] > 0:

                    d[i] = (2*dl[i]*dr[i])/(dl[i]+dr[i])
            
            return d
        
        def delta(q):# superbee
            N = len(q)
            d = np.zeros(N)
            # diferenças com periodicidade
            dl = q - np.roll(q, 1)          # q[j] - q[j-1]
            dr = np.roll(q, -1) - q         # q[j+1] - q[j]
            for i in range (N):
                if dl[i]*dr[i] > 0:
                    r = dr[i]/dl[i]
                    phi_r = max(0, min(1,2*r), min(2,r))
                    d[i] = phi_r * dl[i]
            
            return d
        
        #fluxo a direita e a esqueda em cada célula
        def compute_fluxes(eta, u):
            N = len(eta)
            deta = delta(eta)
            du = delta(u)
            
            F_eta = np.zeros(N)
            F_u   = np.zeros(N)
            
            for j in range(N):
                jp1 = (j + 1) % N   # periodicidade
                
                # Reconstrução MUSCL na interface j (entre j e j+1)
                eta_L = eta[j] + 0.5 * deta[j]
                u_L   = u[j]   + 0.5 * du[j]
                eta_R = eta[jp1] - 0.5 * deta[jp1]
                u_R   = u[jp1] - 0.5 * du[jp1]
                
                # Fluxo de Godunov para o sistema linearizado (|A|=I)
                F_eta[j] = 0.5 * (u_L + u_R - (eta_R - eta_L))
                F_u[j]   = 0.5 * (eta_L + eta_R - (u_R - u_L))
            
            return F_eta, F_u
        
        def muscl(q_eta, q_u):
            F_eta, F_u = compute_fluxes(q_eta, q_u)
            dx = self.dom.dx
            # dq/dt = - (F_j - F_{j-1}) / dx
            new_q_eta = -(F_eta - np.roll(F_eta, 1)) / dx
            new_q_u   = -(F_u   - np.roll(F_u, 1))   / dx
    
            return {
                'new_q_eta' : new_q_eta  ,
                'new_q_u' : new_q_u
            } 

        if modo == "muscl_ssprk22":
            #primeiro estágio ssprk22
            eta_1 = cond_eta + self.dom.dt * muscl(cond_eta, cond_u)['new_q_eta']
            u_1 = cond_u + self.dom.dt * muscl(cond_eta,cond_u)['new_q_u']
            
            #segundo estágio ssprk22
            eta_2 = 0.5*cond_eta + 0.5*eta_1 + 0.5*self.dom.dt*muscl(eta_1, u_1)['new_q_eta']
            u_2 = 0.5*cond_u + 0.5*u_1 + 0.5*self.dom.dt*muscl(eta_1, u_1)['new_q_u']      
            
            return {
                    'eta_final' : eta_2,
                    'u_final': u_2
                }  
           
        elif modo == "muscl_ssprk33":
            #primeiro estágio ssprk33
            eta_1 = cond_eta + self.dom.dt * muscl(cond_eta, cond_u)['new_q_eta']
            u_1 = cond_u + self.dom.dt * muscl(cond_eta,cond_u)['new_q_u']
            
            #segundo estágio ssprk33
            eta_2 = 0.75*cond_eta + 0.25*eta_1 + 0.25*self.dom.dt*muscl(eta_1, u_1)['new_q_eta']
            u_2 = 0.75*cond_u + 0.25*u_1 + 0.25*self.dom.dt*muscl(eta_1, u_1)['new_q_u']      
            
            #terceiro estágio ssprk33
            eta_3 = (1/3)*cond_eta + (2/3)*eta_2 + (2/3)*self.dom.dt*muscl(eta_2, u_2)['new_q_eta']
            u_3 = (1/3)*cond_u + (2/3)*u_2 + (2/3)*self.dom.dt*muscl(eta_2, u_2)['new_q_u']      
            

            return {
                    'eta_final' : eta_3,
                    'u_final': u_3
                } 

    def solucao_numerica(self,
                         solucao_eta:  np.ndarray = None, # condição incial para eta
                         solucao_u: np.ndarray = None, #condição inicial para u
                         tempo: int = None, # tempo de execução do método
                         modo: str = "muscl_ssprk33" # modelo de execução
                         ):
        """Calcula a solução de águas rasas após vários instantes."""

        if tempo is None:
            tempo = self.dom.M
            
        if solucao_eta is None:
            solucao_eta = self.eta_zero()
    
        if solucao_u is None:
            solucao_u = self.u_zero()
          
        if modo == "godunov_euler":
            
            propagacao = self.godunov_euler(eta = solucao_eta, u= solucao_u)
            
            for _ in range(tempo+1):
                
                eta_final = propagacao['eta_final']
                u_final = propagacao['u_final']
                propagacao = self.godunov_euler(eta = eta_final, u= u_final)
       
        elif modo == "muscl_ssprk22":
        
            propagacao = self.muscl_ssprk(solucao_eta,solucao_u, modo = "muscl_ssprk22")
            
            for _ in range(tempo+1):
                eta_final = propagacao['eta_final']
                u_final = propagacao['u_final']            
                propagacao = self.muscl_ssprk(eta_final,u_final, modo = "muscl_ssprk22")

        elif modo == "muscl_ssprk33":
        
            propagacao = self.muscl_ssprk(solucao_eta,solucao_u, modo = "muscl_ssprk33")
            
            for _ in range(tempo+1):
                eta_final = propagacao['eta_final']
                u_final = propagacao['u_final'] 
                propagacao = self.muscl_ssprk(eta_final,u_final, modo = "muscl_ssprk33")

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
                testes: int = 6,
                modo: str = "muscl_ssprk33"
                ):
        self.testes = testes
        self.delta_E = 0
        self.energia_total = 0
        self.dom = dom
        self.modo = modo
    
    def valores_cfl(self):
        from rich.table import Table
        from rich import print
        passo = 1024
        tab = Table(title = " Número de Courant.")
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
    
    def ordem_de_convergencia(self):
            """ Apresenta uma tabela com os erros de aproximação """
            import math
            from tqdm import tqdm
            from rich import print
            from rich.table import Table
            vetor_erro = []
            tab = Table(title = f"Ordem de convergência para eta para modelo {self.modo}.")
            tab.add_column("i", justify = "center")
            tab.add_column("N", justify = "center")
            tab.add_column("M", justify = "center")
            tab.add_column("Courant", justify = "center")
            tab.add_column("Erro", justify = "center")
            tab.add_column("Ordem", justify = "center", style = "red")
            N_ref = self.dom.N
            M_ref = self.dom.M
            for j in tqdm(range(self.testes)):
                
                domi = dominio.Dominio(N = int(N_ref*4**(j-3)),  M = int(M_ref*4**(j-3)))
                s = SolucaoAguasRasas(domi)
                vetor_erro += [max(np.abs(s.solucao_analitica_eta()-s.solucao_numerica(modo =self.modo)['eta']))] # erro na norma infinito
                #vetor_erro += [np.mean(np.abs(s.solucao_analitica_eta()-s.solucao_numerica(modo =self.modo)['eta']))] # erro na norma 1

                if j == 0:
                    tab.add_row(f"{j+1}",f"{domi.N}", f"{domi.M}", f"{s.calculo_cfl()}", f"{vetor_erro[j]:.4e}", None )
                else:
                    tab.add_row(f"{j+1}",f"{domi.N}", f"{domi.M}", f"{s.calculo_cfl()}", f"{vetor_erro[j]:.4e}", f"{math.log(abs(vetor_erro[j-1]/vetor_erro[j]))/math.log(4):.4e}" )
    
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
    

if __name__ == "__main__":
    import dominio
    import matplotlib.pyplot as plt
    import construtor_de_graficos as cdg
    import numpy as np

    #discretizacao = "godunov_euler"
    discretizacao = "muscl_ssprk33"
    #dom = Dominio(N=1024, M = 512) #cfl = 0.5
    #dom = Dominio(N=1024, M = 320) #cfl = 0.8
    dom = Dominio(N=1024, M=256) #cfl = 1
    sol = SolucaoAguasRasas(dom)
    cfl = sol.calculo_cfl()
    val = Validacao(dom, modo = discretizacao, testes = 5)
    
    
    op = 6
    it = 128


    if op == 6: # teste da ordem de convergênia da solução numérica
        val.ordem_de_convergencia()

    elif op == 5: # construção de valores de cfl para teste
        val.valores_cfl()

    elif op == 4: # teste da evolução da energia do sistema
        val.evolucao_da_energia(modo = "leapfrog")

    elif op == 3:#teste do calculo da energia do sistema
        print(f'variação de energia = {val.variacao_de_energia()}')
        print(f'cfl = {sol.calculo_cfl()}')
   
    elif op == 2: # teste da solução numerica para η
        for i in range(it):

            solucao = sol.solucao_numerica(modo = discretizacao, tempo = i)
            z = sol.solucao_analitica_eta(tempo = i)
            y = solucao['eta']
            e_22 = np.linalg.norm(z-y)
                        
            if e_22 > 1:
                print(f"considerando {discretizacao} o erro { e_22}")
                print("erro muito grande.")
                break
            plt.clf()
            plt.xlim(dom.L0, dom.L) # x limit
            plt.plot(dom.x, y, label = 'Solução numérica ' )
            plt.plot(dom.x, z, label = 'Solução Analítica')
            plt.title(f'Execução {i+1} de {it} do modelo {discretizacao} com cfl = {cfl}.')
            plt.legend()

            #plt.show(block = False)
            plt.pause(0.001)

        plt.show()               

    elif op == 1: # teste da cfl
        print(f'dt = {dom.dt}')
        print(f'dx = {dom.dx}')
        print(f'cfl = {sol.calculo_cfl()}')

    elif op == 0: # teste da solução analítica para η
        y = sol.solucao_analitica_eta()

        graf = cdg.Grafico2d(dom.x,y)
        graf.plot2d()

    elif op == -1: #lixo 
        """
        def ssprk33(self,
        cond_eta: np.ndarray = None,
        cond_u: np.ndarray = None,
        t: int = None):
        Avança uma unidade de tempo usando Runge-Kutta 33 
      
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
    """