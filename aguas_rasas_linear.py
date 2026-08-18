#Linear shallow water 
import numpy as np # numerical package
import math # mathematical package
from rich.traceback import install # to help debug
from rich import print # create beautiful tables
from rich.console import Console #to help on debug 
from tqdm import tqdm # para show execute time 
import random # generate noise in samples
import matplotlib.pyplot as plt # to create graphics 
console = Console() # to enhance print() 
install() # to help debug

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
            tempo = self.dom.M   

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

    def malha_c(self,
                cond_eta: np.ndarray = None,# lacated in center of cell x_i
                cond_u: np.ndarray = None # located in edge of cell x_{i+0.5}
                ):
       
        def diff_u(vet):
            return (np.roll(vet,1) - vet)/self.dom.dx

        def diff_eta(vet):
            return (vet - np.roll(vet,-1))/self.dom.dx

        #first stage of ssprk33  
        eta_1 = cond_eta + self.dom.dt*diff_u(cond_u)
        u_1 = cond_u + self.dom.dt*diff_eta(cond_eta)

        #second stage of ssprk33
        eta_2 = 0.75*cond_eta +0.25*eta_1+0.25*self.dom.dt*diff_u(u_1)
        u_2 = 0.75*cond_u +0.25*u_1+0.25*self.dom.dt*diff_eta(eta_1)

        #tird stage of ssprk33
        eta_3 = (1/3)*cond_eta +(2/3)*eta_2 + (2/3)*self.dom.dt*diff_u(u_2)
        u_3 = (1/3)*cond_u +(2/3)*u_2 + (2/3)*self.dom.dt*diff_eta(eta_2)


        return {
                'eta_final' : eta_3,
                'u_final': u_3
            }
    
    def solucao_numerica(self,
                         solucao_eta:  np.ndarray = None, # condição incial para eta
                         solucao_u: np.ndarray = None, #condição inicial para u
                         tempo: int = None, # tempo de execução do método
                         modo: str = "malha_c" # modelo de execução
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

        elif modo == "malha_c":
        
            propagacao = self.malha_c(solucao_eta,solucao_u)
            
            for _ in range(tempo+1):
                eta_final = propagacao['eta_final']
                u_final = propagacao['u_final'] 
                propagacao = self.malha_c(eta_final,u_final)

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
                #vetor_erro += [max(np.abs(s.solucao_analitica_eta()-s.solucao_numerica(modo =self.modo)['eta']))] # erro na norma infinito
                vetor_erro += [np.mean(np.abs(s.solucao_analitica_eta()-s.solucao_numerica(modo =self.modo)['eta']))] # erro na norma 1

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
                 n_amostras: int = 2,
                 standard_deviation: float = 0.0005,
                 condicao: str = "condicao_paper",
                 ruido: bool = False,
                 modo: str = "malha_c",
                 Delta_x: float = 0.09,
                 first_sample: float = 0.2
                 ):
        
        self.Delta_x = Delta_x
        self.first_sample = first_sample

        self.n_amostras = n_amostras
        self.dom = dom
        self.standard_deviation = standard_deviation
        self.condicao = condicao
        self.ruido = ruido
        self.sol = SolucaoAguasRasas(self.dom, condicao=self.condicao)
        self.modo = modo
        self.matriz_com_amostras = None
        self.matriz_com_amostras_ruido = None
        self.vetor_custo = []
        #self.vetor_ruido = [random.uniform(0, 0.005) for _ in range(self.dom.N)]
        self.matriz_ruido = np.array([[random.gauss(0, self.standard_deviation) for _ in range(self.dom.M)] for _ in range(self.n_amostras)]) # matriz de ordem n_amostrasxM
        #print(f'tamanho da matrix {self.matriz_ruido.shape}')
        self.E = np.abs(np.mean(np.sum(self.matriz_ruido, axis=1)))

        
        self.matriz_de_amostras_ruido()
        self._print_passos_done = True
                
    def construtor_passos(self,
                        print_info: bool = False):
        
        #FIXME:
        #! Por algum motivo o construtor de passos retorna erro quando usamos delta_x = 0.1 
        #! Investigar após a reunião com o professor pedro
        
        janela_de_observacao = self.dom.x[(self.dom.x>0) & (self.dom.x<2)]
        
        # print_info = True

        if print_info:        
            print('---------------------------')
            print(f'Delta x informado {self.Delta_x}')
            print(f'x0 informado {self.first_sample}')
            print('---------------------------')
            print('')

        if (self.first_sample < 0) or (self.first_sample > 2) or (self.Delta_x < 0) or (self.Delta_x > 2): #eliminar possibilidades absurdas
        
            print('Valores incompatíves com a janela de observação e será adotado')
            self.first_sample = janela_de_observacao[1] # retorna o primeiro valor não nulo da janela de observação
            self.Delta_x = self.dom.dx # retorna o dx ótimo de assimilação

        else:
            if np.any(np.isclose(janela_de_observacao, self.first_sample)): #x_0 é compatível com a discretização
                x_ultimo = self.first_sample + (self.n_amostras-1)*self.Delta_x
                #print(f'ultima amostras = {x_ultimo}')
                if np.any(np.isclose(janela_de_observacao,x_ultimo)):#x_j é compatível com a discretização
                    pass #os dados informados são compatíves com discretização

                else: #x_0 é compatível com a discretização mas xj não
                    if self.Delta_x >= self.dom.dx:# se Delta_x> dx basta adaptar o Delta_x ao dx
                        Delta_x_local = np.floor(self.Delta_x/self.dom.dx)*self.dom.dx if (np.floor(self.Delta_x/self.dom.dx) != 0) else self.dom.dx
                        x_ultimo = self.first_sample + (self.n_amostras-1)*Delta_x_local
                        if np.any(np.isclose(janela_de_observacao,x_ultimo)):# se a adaptação não ultrapaçar a janela ok
                            self.Delta_x = Delta_x_local
                        else:# se a adaptação ultrapassar a janela
                            Delta_x_max = (2 - self.first_sample)/self.n_amostras #maior delta_x para a primeira amostra fornecida
                            if Delta_x_max <= self.dom.dx: # testando a posição da primeira amostras
                                self.Delta_x = self.dom.dx
                                self.first_sample = 2 - (self.n_amostras+3)*self.dom.dx
                            else:
                                Delta_x_local = np.floor(Delta_x_max/self.dom.dx)*self.dom.dx if (np.floor(Delta_x_max/self.dom.dx) != 0) else self.dom.dx
                                self.Delta_x = Delta_x_local

                    else:
                        self.Delta_x = self.dom.dx
                        self.first_sample = 2 - (self.n_amostras+3)*self.dom.dx

            else:
                self.first_sample =janela_de_observacao[np.argmin(np.abs(janela_de_observacao - self.first_sample))]
                x_ultimo = self.first_sample + (self.n_amostras-1)*self.Delta_x
                #print(f'ultima amostras = {x_ultimo}')
                if np.any(np.isclose(janela_de_observacao,x_ultimo)):#x_j é compatível com a discretização
                    pass #os dados informados são compatíves com discretização

                else: #x_0 é compatível com a discretização mas xj não
                    if self.Delta_x >= self.dom.dx:# se Delta_x> dx basta adaptar o Delta_x ao dx
                        Delta_x_local = np.floor(self.Delta_x/self.dom.dx)*self.dom.dx if (np.floor(self.Delta_x/self.dom.dx) != 0) else self.dom.dx
                        x_ultimo = self.first_sample + (self.n_amostras-1)*Delta_x_local
                        if np.any(np.isclose(janela_de_observacao,x_ultimo)):# se a adaptação não ultrapaçar a janela ok
                            self.Delta_x = Delta_x_local
                        else:# se a adaptação ultrapassar a janela
                            Delta_x_max = (2 - self.first_sample)/self.n_amostras #maior delta_x para a primeira amostra fornecida
                            if Delta_x_max <= self.dom.dx: # testando a posição da primeira amostras
                                self.Delta_x = self.dom.dx
                                self.first_sample = 2 - (self.n_amostras+3)*self.dom.dx
                            else:
                                Delta_x_local = np.floor(Delta_x_max/self.dom.dx)*self.dom.dx if (np.floor(Delta_x_max/self.dom.dx) != 0) else self.dom.dx
                                self.Delta_x = Delta_x_local

                    else:
                        self.Delta_x = self.dom.dx
                        self.first_sample = 2 - (self.n_amostras+3)*self.dom.dx




        xj = np.array([self.first_sample + i*self.Delta_x for i in range(self.n_amostras)])
        #print(f'vetor xj = {xj}')
        position = [np.where(np.isclose(self.dom.x, xj[i]))[0][0] for i in range(self.n_amostras)]
        passos = [int(p) for p in position]
        

        if print_info:   
            print('---------------------------')
            print(f'Delta x adotado {self.Delta_x}')
            print(f'x0 adotado {self.first_sample}')
            print('---------------------------')
            print('')
            if self.Delta_x > 0.1:
                console.print("[bold red] O texto exige Delta_x < 0.1 para garantir a assimilação [/bold red]")
            self._print_passos_done = False

        
        return {
            'passos' : passos,
            'xj': xj 
        }
    
    def matriz_de_amostras(self):

        """Gera uma matriz contendo as amostras sem perturbação"""
        
        matriz = np.zeros((self.n_amostras, self.dom.M)) # as amostras serão armazenadas em linhas 

        steps = self.construtor_passos()['passos']


        for i in range(self.n_amostras):
            for j in range(self.dom.M):
                soluction = sol.solucao_analitica_eta(tempo=j)   
                matriz[i, j] = soluction[steps[i]]
            
        self.matriz_com_amostras = matriz
        return matriz # retorna uma matriz de ordem n_amostrasxM
        
    def matriz_de_amostras_ruido(self):
        """Gera uma matriz contendo as amostras com perturbação """
        if self.matriz_com_amostras_ruido is None:
            self.matriz_de_amostras()     
        matriz_com_ruido = self.matriz_com_amostras.copy() # gera uma cópia da matriz de amostras para não precisar construir outra
                
        amostras_com_ruido = matriz_com_ruido + self.matriz_ruido
        self.matriz_com_amostras_ruido = amostras_com_ruido
        return amostras_com_ruido # retorna uma matriz de ordem n_amostrasxM

    def forcante(self,
                u: np.ndarray = None,
                eta: np.ndarray = None,
                ): # forçante do método de volumes finitos


        x_j = self.construtor_passos()["passos"]
        #soluction = SolucaoAguasRasas(dom = self.dom)
        #forecast = SolucaoAguasRasas(dom = self.dom)
        forcante = np.zeros((self.dom.N, self.dom.M))   
        if self.ruido: 
            for i in range(self.n_amostras):
                for j  in range(self.dom.M):
                    eta_forecast = sol.solucao_numerica(solucao_eta = eta, solucao_u = u, modo = "malha_c", tempo = j)["eta"]
                    y_j =  sol.solucao_analitica_eta(tempo = j )
                    forcante[x_j[i],j] = eta_forecast[x_j[i]] - (y_j[x_j[i]] + self.matriz_ruido[i, j])
        else:
            for i in range(self.n_amostras): # loop para gerar os pontos amostrais
                for j in range(self.dom.M): # loop para gerar toda a evolução temporal 
                    eta_forecast = sol.solucao_numerica(solucao_eta = eta, solucao_u = u, modo = "malha_c", tempo = j)["eta"] # E volui o problema direto tomando o chute inicial
                    y_j =  sol.solucao_analitica_eta(tempo = j ) # obter as amostras atravez da solução analítica tomando a condição incial verdadeira
                    forcante[x_j[i],j] = eta_forecast[x_j[i]] - y_j[x_j[i]] # calcula a diferença necessária no termo forçante.
            
        return forcante
    
    def grad_analitico(self,
                       phi_n: np.ndarray = None
                    ):
        
        """ recebe o n_ésimo condição inicial e 
        calcula o gradiente analítico."""
        def composta(vetor, j):
            vetor_invertido = vetor[::-1]
            vetor_deslocado = np.roll(vetor_invertido, -(len(vetor)-1)+2*j)
            return vetor_deslocado
        
        eta_zero = self.sol.eta_zero()
        soma = []
        passos_local = self.construtor_passos()['passos']
        for  i in range(self.n_amostras):
            j = passos_local[i]
            soma =+ composta(eta_zero, j)-composta(phi_n, j)
        grad = -(self.n_amostras/4)*(eta_zero-phi_n+(1/self.n_amostras)*soma)

        return grad

    def grad(self,
                cond_eta: np.ndarray = None,
                cond_u: np.ndarray = None
                ):
        #NOTE: I think it's okay.

        fonte = self.forcante(u = cond_u, eta = cond_eta )
        #print(f'Ordem da forçante = {fonte.shape}')
        u_zero_star = np.zeros(self.dom.N)
        eta_zero_star = np.zeros(self.dom.N)

        def diff_u(vet):
            return (np.roll(vet,1) - vet)/self.dom.dx

        def diff_eta(vet):
                    return (vet - np.roll(vet,-1))/self.dom.dx

        #here we have a reverse system, so $\tilde{\delta_t} = \delta_t$.
        for i in reversed(range(self.dom.M-1)):
            k = i+1
            #first stage of ssprk33  
            eta_1 = eta_zero_star - self.dom.dt*(diff_u(u_zero_star)-(1/self.dom.dx) * fonte[:,k])
            u_1 = u_zero_star- self.dom.dt*diff_eta(eta_zero_star)

            #second stage of ssprk33
            eta_2 = 0.75*eta_zero_star + 0.25*eta_1 - 0.25*self.dom.dt*(diff_u(u_1)-(1/self.dom.dx) * fonte[:,k-1])
            u_2 = 0.75*u_zero_star + 0.25*u_1 - 0.25*self.dom.dt*diff_eta(eta_1)

            #second stage of ssprk33
            eta_3 = (1/3)*eta_zero_star +(2/3)*eta_2 - (2/3)*self.dom.dt*(diff_u(u_2)-(1/self.dom.dx) * fonte[:,k-1])
            u_3 = (1/3)*u_zero_star +(2/3)*u_2 - (2/3)*self.dom.dt*diff_eta(eta_2)

            eta_zero_star = eta_3
            u_zero_star = u_3

        

        return {
                'eta_grad' : eta_3,
                'u_grad': u_3
            } 

    def custo_assimilacao(self,
                          eta: np.ndarray = None,
                          ):
        """Retorna o custo de assimilação para cada iteração."""
        steps = self.construtor_passos()['passos'] #gera o indice onde estão as amostras no vetor de assiilação
        diff= np.zeros((self.n_amostras, self.dom.M))# vai receber as diferenças internas do custo
        if self.ruido: #condicional para construir o y_j que são as amostas
            self.matriz_de_amostras_ruido()
            y_j = self.matriz_com_amostras_ruido # é uma matriz de ordem n_amostrasxM
        else:
            self.matriz_de_amostras()
            y_j = self.matriz_com_amostras
        
        for i in range(self.dom.M): # loop para construir a diferença presente no custo
            eta_f = self.solucao_numerica(solucao_eta=eta, solucao_u=np.zeros(self.dom.N), tempo=i)['eta'] # constroi o eta^f dada a condicao tomando u = 0
            # Atualiza solução
            for j in range(self.n_amostras):
                diff[j,i] = (eta_f[steps[j]] - y_j[j,i])**2
        sum_diff = np.sum(diff, axis=0) #retorna um vetor de tamanho self.M com a soma das n_amostras colunas 

        def trapezoidal_rule(x): # integral via regra do trapézio
            s=0
            n = len(x)
            for i in range(1,n-1,1):
                s += x[i]
            return (x[0] + 2*s + x[-1])*self.dom.dt/2

        return 0.5 * trapezoidal_rule(sum_diff ) # retorna a integral numérica do quadrado da diferença        
  
    def diferenca(self,
                  iter: int = 10):
        diferenca = []
        for i in range(iter):
            #diferenca.append(np.abs(np.mean((self.sol.u_zero(dom.x)-self.gradiente_descendente(it = i)))))
            diferenca.append(float(np.linalg.norm(self.sol.u_zero(self.dom.x)-self.gradiente_descendente(it = i)['eta_final'])/np.linalg.norm(self.sol.u_zero(self.dom.x))))

        return diferenca

    def gradiente_descendente(self,
                              it:int = 10):
        """Calculo do gradiente descendente considerando n=it iterações"""
        def reconstruction_error(vet):
            return np.linalg.norm(vet - self.sol.eta_zero())/np.linalg.norm(self.sol.eta_zero())
        from tqdm import tqdm
        solucao_final_eta = np.zeros(self.dom.N) #chute inicial
        solucao_final_u = np.zeros(self.dom.N) #chute inicial
        error = []
        custo = []
        if self.modo == "analitico":
            for _ in tqdm(range(it)):
                solucao_final_eta = solucao_final_eta - 0.1*self.grad_analitico(solucao_final_eta)    
                error.append(reconstruction_error(solucao_final_eta))
                custo.append(self.custo_assimilacao(solucao_final_eta))
        else:            
            for _ in tqdm(range(it)):
                grad = self.grad(cond_eta = solucao_final_eta, cond_u = solucao_final_u)
                solucao_final_eta = solucao_final_eta - 0.1*grad["eta_grad"]
                solucao_final_u = solucao_final_u - 0.1*grad["u_grad"]
                error.append(reconstruction_error(solucao_final_eta))
                custo.append(self.custo_assimilacao(solucao_final_eta))

        return {
                'eta_final' : solucao_final_eta, # eta após it execuções do gradiente descendente con learning rate fixo
                'u_final': solucao_final_u, # u após it execuções do gradiente descendente con learning rate fixo
                'error' : error, # Erro de reconstrução de cada passo do gradiente descendente
                'custo': custo # funcional custo de cada passo do gradiente descendente
            }

    def gradiente_descendente_otimizado(self,
                                it:int = 10):
            """Calculo do gradiente descendente considerando n=it iterações"""
            def reconstruction_error(vet):
                return np.linalg.norm(vet - self.sol.eta_zero())/np.linalg.norm(self.sol.eta_zero())
            from tqdm import tqdm
            from scipy.optimize import line_search
            solucao_final_eta = np.zeros(self.dom.N) #chute inicial
            solucao_final_u = np.zeros(self.dom.N) #chute inicial
            error = []
            custo = []
            alpha = []
            if self.modo == "analitico":
                for _ in tqdm(range(it)):
                    grad = self.grad_analitico(solucao_final_eta)
                    otimi = line_search(self.custo_assimilacao, self.grad_analitico, solucao_final_eta, -grad)
                    solucao_final_eta = solucao_final_eta - otimi[0]*grad   
                    error.append(reconstruction_error(solucao_final_eta))
                    custo.append(self.custo_assimilacao(solucao_final_eta))
                    alpha.append(otimi[0])
            else:
                
                
                for i in tqdm(range(it)):
                    def grad_eta(eta): # função para retornar apenas o grad_eta utlizado na otimização
                        return self.grad(cond_eta = eta, cond_u = solucao_final_u)['eta_grad']
                    grad_eta_local = grad_eta(eta = solucao_final_eta) # gera a direção de decaimento
                    grad_u = self.grad(cond_eta = solucao_final_eta, cond_u = solucao_final_u)['u_grad']         
                    otimi = line_search(self.custo_assimilacao, grad_eta, solucao_final_eta, -grad_eta_local ) #gera a otimização do passo do gradiente descendente
                    if otimi[0] is None: # garante que o gradiente irá funcionar mesmo se não houver otimização do passo do gradiente descendente
                        alpha = 0.1
                        print(f"Não houve otimização do passo na iteração {i}")
                    else:
                        alpha = otimi[0]
                    solucao_final_eta = solucao_final_eta - alpha*grad_eta_local
                    solucao_final_u = solucao_final_u - 0.1*grad_u
                    error.append(reconstruction_error(solucao_final_eta))
                    custo.append(self.custo_assimilacao(solucao_final_eta))
                    alpha.append(alpha)

            return {
                    'eta_final' : solucao_final_eta, # eta após it execuções do gradiente descendente con learning rate fixo
                    'u_final': solucao_final_u, # u após it execuções do gradiente descendente con learning rate fixo
                    'error' : error, # Erro de reconstrução de cada passo do gradiente descendente
                    'custo': custo, # funcional custo de cada passo do gradiente descendente
                    'alpha': alpha, # passo do gradiente descendente para ser aproveitado posteriormente
                }
        
if __name__ == "__main__":
    import dominio
    import matplotlib.pyplot as plt
    import construtor_de_graficos as cdg
    import numpy as np
    import hashlib
    import json
    from pathlib import Path

    ###opção
    op = 14
    iteracoes = 2**8

    #### Variáveis
    # N=1025; M = 513 #cfl = 0.5
    # N=1024; M = 320 #cfl = 0.8
    N=512;  M = 160 #cfl = 0.8
    # N=1025; M=257   #cfl = 1
    amos = 2
    ruido = False
    first_sample = 1 # paper uses first_sample = 0.2
    Delta_x =  0.09 # paper uses Delta_x = 0.09 end Delta_x = 0.375 for counter-example
    #discretizacao = "godunov_euler"
    #discretizacao = "muscl_ssprk33"
    discretizacao = "malha_c"
    modo = "malha_c"
    #modo = "analitico"
    save = True
    
    ###objetos

    #dom = Dominio(N=1025, M = 513) #cfl = 0.5
    #dom = Dominio(N=1024, M = 320) #cfl = 0.8
    dom = Dominio(N = N, M = M) #cfl = 0.8
    #dom = Dominio(N=1025, M=257) #cfl = 1
    sol = SolucaoAguasRasas(dom)
    cfl = sol.calculo_cfl()
    val = Validacao(dom, modo = discretizacao, testes = 6)
    ass = Assimilacao(dom, modo= modo, n_amostras = amos,
                       ruido=ruido,
                       first_sample = first_sample, 
                       Delta_x =  Delta_x ) 



    def save_file():


        params_dict = {
                "model" : 'swe_l',
                "M" : M,
                "N" : N,
                "it" : iteracoes,
                "noise" : ruido,
                "first_sample" : first_sample,
                "Delta_x" :  Delta_x,
                "discratization": discretizacao,
                "grad": modo 
            }
        params_string = json.dumps(params_dict, sort_keys=True) # create an javascript string

        hash_code = hashlib.md5(params_string.encode('utf-8')).hexdigest() # criate a name to the file

        folder = Path("./data") # identify the folder

        save_path = folder / f"{hash_code}.npz" 

        np.savez_compressed(
            save_path,
            alphas=None,
            trajetoria=None,
            custo_final=None,

        )


    if op == 20: #validação teorica
        ass10 = Assimilacao(dom, modo= modo, n_amostras = 2, ruido=False, first_sample = 0.2, Delta_x =  0.09 ) 
        result10 = ass10.gradiente_descendente(it=iteracoes)['error']
        ass11 = Assimilacao(dom, modo= modo, n_amostras = 2, ruido=True, first_sample = 0.2, Delta_x =  0.09 ) 
        result11 = ass11.gradiente_descendente(it=iteracoes)['error']
        ass20 = Assimilacao(dom, modo= modo, n_amostras = 3, ruido=False, first_sample = 0.2, Delta_x =  0.09 ) 
        result20 = ass20.gradiente_descendente(it=iteracoes)['error']
        ass21 = Assimilacao(dom, modo= modo, n_amostras = 3, ruido=True, first_sample = 0.2, Delta_x =  0.09 ) 
        result21 = ass21.gradiente_descendente(it=iteracoes)['error'] 
        ass30 = Assimilacao(dom, modo= modo, n_amostras = 4, ruido=False, first_sample = 0.2, Delta_x =  0.09 ) 
        result30 = ass30.gradiente_descendente(it=iteracoes)['error'] 
        ass31 = Assimilacao(dom, modo= modo, n_amostras = 4, ruido=True, first_sample = 0.2, Delta_x =  0.09 ) 
        result31 = ass31.gradiente_descendente(it=iteracoes)['error'] 
        ass40 = Assimilacao(dom, modo= modo, n_amostras = 5, ruido=False, first_sample = 0.2, Delta_x =  0.09 ) 
        result40 = ass40.gradiente_descendente(it=iteracoes)['error'] 
        ass41 = Assimilacao(dom, modo= modo, n_amostras = 5, ruido=True, first_sample = 0.2, Delta_x =  0.09 ) 
        result41 = ass41.gradiente_descendente(it=iteracoes)['error'] 
        ass50 = Assimilacao(dom, modo= modo, n_amostras = 6, ruido=False, first_sample = 0.2, Delta_x =  0.09 ) 
        result50 = ass50.gradiente_descendente(it=iteracoes)['error']
        ass51 = Assimilacao(dom, modo= modo, n_amostras = 6, ruido=True, first_sample = 0.2, Delta_x =  0.09 ) 
        result51 = ass51.gradiente_descendente(it=iteracoes)['error']

        plt.ylabel(fr"$|\\phi^t(x) - \\phi^n(x)|$")
        plt.xlabel('Número de iterações')
        #ax.set_xscale('log')
        plt.yscale('log')
        #plt.ylabel('J(x)')
        plt.scatter([i+1 for i in range(iteracoes)], result10, lw = 0.5, label = '2 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], result11, lw = 0.5, label = '2 amostras com ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], result20, lw = 0.5, label = '3 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], result21, lw = 0.5, label = '3 amostras com ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], result30, lw = 0.5, label = '4 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], result31, lw = 0.5, label = '4 amostras com ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], result40, lw = 0.5, label = '5 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], result41, lw = 0.5, label = '5 amostras com ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], result50, lw = 0.5, label = '6 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], result51, lw = 0.5, label = '6 amostras com ruido' )
        plt.title(f'Erro de reconstrução após {iteracoes} Delta x =  {ass.Delta_x}.')
        plt.legend()
        plt.show()
        

    elif op == 15: #teste de otimizacao

        erro_otimizado = ass.gradiente_descendente_otimizado(it=iteracoes)['error']
        erro = ass.gradiente_descendente(it=iteracoes)['error']

        plt.ylabel('J^(n)')
        plt.xlabel('Número de iterações')
        plt.yscale('log')
        plt.scatter([i+1 for i in range(iteracoes)], erro_otimizado , lw = 0.5, label = 'erro otimizado' )
        plt.scatter([i+1 for i in range(iteracoes)], erro , lw = 0.5, label = 'erro fixo' )
        plt.title(f'Convergencia do custo após {iteracoes} iterações considerando Δx =  {ass.Delta_x}.')
        plt.legend()
        plt.show()
        
    elif op == 14: # construir o gráfico do convergencia J^(n)/J^(0)do custo
        ass2 = Assimilacao(dom, modo= modo, n_amostras = 2, ruido=ruido, first_sample = first_sample, Delta_x =  Delta_x )
        ass3 = Assimilacao(dom, modo= modo, n_amostras = 3, ruido=ruido, first_sample = first_sample, Delta_x =  Delta_x )
        ass4 = Assimilacao(dom, modo= modo, n_amostras = 4, ruido=ruido, first_sample = first_sample, Delta_x =  Delta_x )
        ass5 = Assimilacao(dom, modo= modo, n_amostras = 5, ruido=ruido, first_sample = first_sample, Delta_x =  Delta_x )
        ass6 = Assimilacao(dom, modo= modo, n_amostras = 6, ruido=ruido, first_sample = first_sample, Delta_x =  Delta_x )
        gd2 = ass2.gradiente_descendente(it=iteracoes)
        gd3 = ass3.gradiente_descendente(it=iteracoes)
        gd4 = ass4.gradiente_descendente(it=iteracoes)
        gd5 = ass5.gradiente_descendente(it=iteracoes)
        gd6 = ass6.gradiente_descendente(it=iteracoes)
        gd2_otimizado = ass2.gradiente_descendente_otimizado(it=iteracoes)
        gd3_otimizado = ass3.gradiente_descendente_otimizado(it=iteracoes)
        gd4_otimizado = ass4.gradiente_descendente_otimizado(it=iteracoes)
        gd5_otimizado = ass5.gradiente_descendente_otimizado(it=iteracoes)
        gd6_otimizado = ass6.gradiente_descendente_otimizado(it=iteracoes)
        

        if save:
            info = f"""
            Foram armazendos os vetores custos de assimilação.
            Parâmetros utilizados:
                model : swe_l_op_14,
                M : {M},
                N : {N},
                it : {iteracoes},
                noise : {ruido},
                first_sample : {first_sample},
                Delta_x :  {Delta_x},
                discratization: {discretizacao},
                grad: {modo}
            """

            params_dict = {
                "model" : 'swe_l_op_14',
                "M" : M,
                "N" : N,
                "it" : iteracoes,
                "noise" : ruido,
                "first_sample" : first_sample,
                "Delta_x" :  Delta_x,
                "discretization": discretizacao,
                "grad": modo 
            }
    
            params_string = json.dumps(params_dict, sort_keys=True) # create an javascript string
    
            hash_code = hashlib.md5(params_string.encode('utf-8')).hexdigest() # criate a name to the file
    
            folder = Path("./data") # identify the folder
    
            save_path = folder / f"{hash_code}.npz" 
    
            np.savez_compressed(
                save_path,
                info = info,
                gd2 = gd2,
                gd3 = gd3,
                gd4 = gd4,
                gd5 = gd5,
                gd6 = gd6,
                gd2_otimizado = gd2_otimizado,
                gd3_otimizado = gd3_otimizado,
                gd4_otimizado = gd4_otimizado,
                gd5_otimizado = gd5_otimizado,
                gd6_otimizado = gd6_otimizado,
                
            )
            
        '''plt.ylabel('J^(n)/J^(0)')
        plt.xlabel('Número de iterações')
        plt.yscale('log')
        plt.scatter([i+1 for i in range(iteracoes)], erro2/erro2[0] , lw = 0.5, label = '2 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], erro3/erro3[0] , lw = 0.5, label = '3 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], erro4/erro4[0] , lw = 0.5, label = '4 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], erro5/erro5[0] , lw = 0.5, label = '5 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], erro6/erro6[0] , lw = 0.5, label = '6 amostras sem ruido' )
        plt.title(f'Convergencia do custo após {iteracoes} iterações considerando Δx =  {ass2.Delta_x}.')
        plt.legend()
        plt.show()'''

    elif op == 13: # construir o gráfico do erro de reconstrução da condição incial
        ass2 = Assimilacao(dom, modo= modo, n_amostras = 2, ruido=ruido, first_sample = first_sample, Delta_x =  Delta_x )
        ass3 = Assimilacao(dom, modo= modo, n_amostras = 3, ruido=ruido, first_sample = first_sample, Delta_x =  Delta_x )
        ass4 = Assimilacao(dom, modo= modo, n_amostras = 4, ruido=ruido, first_sample = first_sample, Delta_x =  Delta_x )
        ass5 = Assimilacao(dom, modo= modo, n_amostras = 5, ruido=ruido, first_sample = first_sample, Delta_x =  Delta_x )
        ass6 = Assimilacao(dom, modo= modo, n_amostras = 6, ruido=ruido, first_sample = first_sample, Delta_x =  Delta_x )
        erro2 = ass2.gradiente_descendente(it=iteracoes)['error']
        erro3 = ass3.gradiente_descendente(it=iteracoes)['error']
        erro4 = ass4.gradiente_descendente(it=iteracoes)['error']
        erro5 = ass5.gradiente_descendente(it=iteracoes)['error']
        erro6 = ass6.gradiente_descendente(it=iteracoes)['error']

        plt.ylabel("||(phi^t(x) - phi^n(x))||/||phi^t(x)||")
        plt.xlabel('Número de iterações')
        plt.yscale('log')
        #plt.ylabel('J^(n)/J^(0)')
        plt.scatter([i+1 for i in range(iteracoes)], erro2/erro2[0] , lw = 0.5, label = '2 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], erro3/erro3[0] , lw = 0.5, label = '3 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], erro4/erro4[0] , lw = 0.5, label = '4 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], erro5/erro5[0] , lw = 0.5, label = '5 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], erro6/erro6[0] , lw = 0.5, label = '6 amostras sem ruido' )
        plt.title(f'Erro de reconstrução após {iteracoes} ietrações considerando $Delta_x = $ {ass2.Delta_x}.')
        plt.legend()
        plt.show()

    elif op == 12: # teste do custo J^(n) de assimilação
        
        ass2 = Assimilacao(dom, modo= modo, n_amostras = 2, ruido=False, first_sample = 0.2, Delta_x =  0.09 )
        ass3 = Assimilacao(dom, modo= modo, n_amostras = 3, ruido=False, first_sample = 0.2, Delta_x =  0.09 )
        ass4 = Assimilacao(dom, modo= modo, n_amostras = 4, ruido=False, first_sample = 0.2, Delta_x =  0.09 )
        ass5 = Assimilacao(dom, modo= modo, n_amostras = 5, ruido=False, first_sample = 0.2, Delta_x =  0.09 )
        ass6 = Assimilacao(dom, modo= modo, n_amostras = 6, ruido=False, first_sample = 0.2, Delta_x =  0.09 )
        custo2 = ass2.gradiente_descendente(it=iteracoes)['custo']
        custo3 = ass3.gradiente_descendente(it=iteracoes)['custo']
        custo4 = ass4.gradiente_descendente(it=iteracoes)['custo']
        custo5 = ass5.gradiente_descendente(it=iteracoes)['custo']
        custo6 = ass6.gradiente_descendente(it=iteracoes)['custo']

        #plt.ylabel(fr"$|\\phi^t(x) - \\phi^n(x)|$")
        plt.xlabel('Número de iterações')
        plt.yscale('log')
        plt.ylabel('J^(n)')
        plt.scatter([i+1 for i in range(iteracoes)], custo2 , lw = 0.5, label = '2 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], custo3 , lw = 0.5, label = '3 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], custo4 , lw = 0.5, label = '4 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], custo5 , lw = 0.5, label = '5 amostras sem ruido' )
        plt.scatter([i+1 for i in range(iteracoes)], custo6 , lw = 0.5, label = '6 amostras sem ruido' )
        plt.title(f'Custo de assimilação após {iteracoes} Δx =  {ass2.Delta_x}.')
        plt.legend()
        plt.show()

    elif op == 11: # teste gradiente numerico
        modo1 = "malha_c"
        ass_local = Assimilacao(dom, modo= modo1, n_amostras = amos,
                               ruido=ruido,
                               first_sample = first_sample, 
                               Delta_x =  Delta_x  )
        result = ass_local.gradiente_descendente(it=iteracoes)
        caso = ass_local.construtor_passos()
        passos = caso['xj']
        grf = True
        #ploting the graph
        #plt.clf()
        if grf:

            plt.ylim(-0.025, 0.06) # y limit
            plt.xlim(-1.5, 1.5) # x limit
            plt.plot(dom.x, result['eta_final'], label = 'phi^(f)(x) assimilada' )
            plt.plot(dom.x, sol.eta_zero(dom.x), label = 'phi^(t)(x) realidade')
            for px in passos:# destacar os pontos de amostragem
                plt.plot([px, px], [-0.001, 0.001], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            if ruido:
                plt.title(f'Execução de {iteracoes} iterações utilizando {amos} amostras com Delta x =  {ass.Delta_x}. com ruido'  )
            else:
                plt.title(f'Execução de {iteracoes} iterações utilizando {amos} amostras com Delta x =  {ass.Delta_x}. sem ruido')
            plt.legend()
            plt.pause(0.9)
            #plt.savefig('assimilacao.png')
            plt.show()
        else:

            plt.xlabel('Número de iterações')
            #ax.set_xscale('log')
            plt.yscale('log')
            plt.ylabel('|phi^(t)(x) - phi^(n)(x)|')
            plt.scatter([i+1 for i in range(iteracoes)], result['error'], lw = 0.5, label = 'Erro em cada iteração' )
            plt.title(fr'Execução de {iteracoes} iterações utilizando {amos} amostras com Delta x =  {ass.Delta_x}.')
            plt.legend()
            #plt.savefig('custo_de_assimilacao.png')
            plt.show()

    elif op == 10: # teste gradiente analítico
        modo1 = "analitico"
        ass_local = Assimilacao(dom, modo= modo1, n_amostras = amos,
                               ruido=ruido,
                               first_sample = first_sample, 
                               Delta_x =  Delta_x  )
        caso = ass_local.construtor_passos()
        passos = caso['xj']
        for j in range(iteracoes):
            if math.log2(j+1).is_integer():
                #diff = np.linalg.norm(ass.gradiente_descendente(it=j)-sol.eta_zero(dom.x)) 
                #ploting the graph
                plt.clf()
                plt.ylim(-0.025, 0.06) # y limit
                plt.xlim(-2.3, 2.3) # x limit
                plt.plot(dom.x, ass_local.gradiente_descendente(it=j)['eta_final'], label = 'phi^(f)(x) assimilada' )
                #plt.plot(dom.x, sol.eta_zero(dom.x), label = f'$\phi^{{(t)}}(x)$ realidade' f'\nDiff = : {diff:.2e}' )
                plt.plot(dom.x, sol.eta_zero(dom.x), label = f'phi^(t)(x) realidade' )
                for px in passos:# destacar os pontos de amostragem
                    plt.plot([px, px], [-0.001, 0.001], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                plt.title(f'Execução {j+1} de {iteracoes} utilizando {amos} amostras com Δ x =  {ass.Delta_x}.')
                plt.legend()
                plt.pause(0.9)
  
        plt.show()
    
    elif op == 9: #teste gradiente step by step
        iteracoes_local = 16
        ass_local = Assimilacao(dom, modo="analitico", n_amostras = amos,
                       ruido=ruido,
                       first_sample = 0.2,
                       Delta_x = 0.09  )
        for i in range(iteracoes_local):

            y = ass_local.gradiente_descendente(it=i)
            z = sol.eta_zero(dom.x)
            graf = cdg.Grafico2d(dom.x, y1=y,y2=z, y1_name="assimilação", y2_name="realidade", title=f"{i} iterações",  fontsize=16)
            graf.plot2d()
            plt.clf()

    elif op == 8: #imprimir os passos das amostras
    
        caso1 = ass.construtor_passos()
        print(f'passos de amostragem {caso1['passos']}')
        print(f'Vetor xj = {caso1['xj']}')

    elif op == 7: #Grafico de todas as amostras
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        passo = ass.construtor_passos(print_info= False)["passos"]

        matriz = ass.matriz_de_amostras()      # shape: (self.dom.N, self.dom.M)
        t = dom.t                              # array de shape (M,)
        x_posicoes = dom.x                     # pode substituir por self.dom.x se existir
            

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
            
        # Itera sobre todas as N linhas (cada uma corresponde a uma posição x)
        for j in range(amos):
            i = passo[j]
            x_fixo = x_posicoes[i]                # valor de x para a linha i
            curva = matriz[j, :]                  # valores no tempo (vetor de tamanho M)
            ax.plot(np.full_like(t, x_fixo),      # eixo X constante
                    t,                            # eixo Y = tempo
                    curva,                        # eixo Z = amplitude
                    linewidth=1,                  # linha fina
                    alpha=0.7,                    # transparência
                    color='blue')                 # cor única (pode variar com colormap)
        
        # Configurações dos eixos
        ax.set_title(f'Amostras considerando a primeira amostra em {ass.first_sample} e Delta_x = {ass.Delta_x} ', fontsize=14, fontweight='bold')
        ax.set_xlabel('Posição x')
        ax.set_ylabel('Tempo t')
        ax.set_zlabel(r'$y^(t)(x_j,t)$')
        
        # Força o eixo X a mostrar todo o intervalo [-4, 4]
        ax.set_xlim(0, dom.L)
        ax.set_xticks(np.linspace(0, dom.L, 9))   # marcas -4, -3, ..., 4
        
        # Ajusta o ângulo de visão
        ax.view_init(25, -60)
        
        # Rotaciona o rótulo do eixo Z para horizontal (opcional)
        zlabel = ax.zaxis.label
        zlabel.set_rotation(0)
        
        plt.show()
    
    elif op == 6: #teste da ordem de convergênia da solução numérica
        val.ordem_de_convergencia()

    elif op == 5: #construção de valores de cfl para teste
        val.valores_cfl()

    elif op == 4: #teste da evolução da energia do sistema
        val.evolucao_da_energia(modo = "leapfrog")

    elif op == 3: #teste do calculo da energia do sistema
        print(f'variação de energia = {val.variacao_de_energia()}')
        print(f'cfl = {sol.calculo_cfl()}')
   
    elif op == 2: #teste da solução numerica para η
        for i in range(iteracoes):

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
            plt.title(f'Execução {i+1} de {iteracoes} do modelo {discretizacao} com cfl = {cfl}.',  fontsize=16)
            plt.legend()

            #plt.show(block = False)
            plt.pause(0.01)

        plt.show()               

    elif op == 1: #teste da cfl
        print(f'dt = {dom.dt}')
        print(f'dx = {dom.dx}')
        print(f'cfl = {sol.calculo_cfl()}')

    elif op == 0: #teste da solução analítica para η
        y = sol.solucao_analitica_eta()

        graf = cdg.Grafico2d(dom.x,y)
        graf.plot2d()

    elif op == -1: #crash course of .npz
        #tutorial from https://www.youtube.com/watch?v=7uNSopXdAnc
        
        # create sample arrays
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])

        # save to uncompressed .npz file
        np.savez('data/data2.npz', array1 = arr1, array2 = arr2)

        #recovering the arrays
        with np.load('data/data2.npz') as data:
            a1 = data['array1']
            a1 = data['array2']

       

        # View all array names in the archive
        print('arrays in the archive', data.files)

        #print(a1)

        #converting the file to dictionary 
        #the archive become manageable
        with np.load('data/data2.npz') as data1:
            dados_dict = dict(data1)

        #editing one specific term
        dados_dict['array1'][0] = -1
        #print(dados_dict['array1'])

        #add a new array
        mat1 = np.ones((5,6))
        dados_dict['mat3'] = mat1
        
        # replace an array completely
        dados_dict['array2'] = np.array([1,1,1])

        #print(dados_dict['mat3'])

        #* deleting an array
        #del dados_dict['array2']
        
        print

        #! Always replace the archive with the new data
        np.savez('data/data2.npz', **dados_dict)

        with np.load('data/data2.npz') as data2:
            print('file in final', data2.files)
                         
    elif op == -2: #lixo 
        '''
        ###################### versão 11/08/2026 ###########################
        ########################### op == 13  ##############################


                samples = 5
        sd_increments = 10        
        desvio = []
        results = np.zeros((sd_increments , samples))
        for i in range(sd_increments):
            sd = 10**(-3)*i*5
            desvio.append(sd)
            for j in range(samples):
                ass_local = Assimilacao(dom, modo = modo, n_amostras = j+2, ruido=ruido, standard_deviation = sd)
                erro = ass_local.gradiente_descendente(it = iteracoes)['error']
                results[i,j] = erro[-1]
            print(f'Encerrado para o {sd_increments}º incremento')
        # Nomes das colunas (rótulos para a legenda)
        nomes_amostras = ['2 amostras', '3 amostras', '4 amostras', '5 amostras', '6 amostras']

        # Cria a figura
        plt.figure(figsize=(10, 6))

        # Plota cada coluna como uma curva separada
        for i, nome in enumerate(nomes_amostras):
            plt.semilogy(desvio, results[:, i], marker='o', label=nome)  # escala log no eixo y

        # Personalização
        plt.xlabel('Desvio padrão')
        plt.ylabel("|phi^t(x) - phi^n(x)|")
        plt.title(f'Erro de reconstrução após {iteracoes} iterações')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()

        # Ajusta os ticks do eixo x para os valores exatos
        plt.xticks(desvio, rotation=45)

        plt.tight_layout()
        plt.show()

        ###################### versão 30/08/2026 ###########################
        ########################### forçante  ##############################
        def forcante(self,
                        u: np.ndarray = None,
                        eta: np.ndarray = None,
                        passos: np.ndarray = None
                        ): # forçante do método de volumes finitos
        
                #! Até o presente momento este código está errado
                
                forcante = np.zeros(self.dom.N)   
                if self.ruido:
                    for j , passos in enumerate(self.passos):
        
                        eta_forecast = sol.solucao_numerica(solucao_eta = eta, solucao_u = u, modo = "muscl_ssprk33", tempo = passos)["eta"]
                        y_j =  self.matriz_com_amostras_ruido[:, j]
                        forcante += eta_forecast - y_j
                else:
                    for j , passos in enumerate(self.passos):
        
                        eta_forecast = sol.solucao_numerica(solucao_eta = eta, solucao_u = u, modo = "muscl_ssprk33", tempo = passos)["eta"]
                        y_j =  self.matriz_com_amostras[:, j]
                        forcante += eta_forecast - y_j
                
                return -forcante/self.dom.dx 
        
        '''


        """
        ###################### versão 30/08/2026 ###########################
        ###################### gradiente numérico ##########################
        
        
        
        #construindo as inclinações delta_j
                def delta(q):# van_leer
                    N = len(q)
                    d = np.zeros(N)
                    # diferenças com periodicidade
                    dl = q - np.roll(q, 1)          # q[j] - q[j-1]
                    dr = np.roll(q, -1) - q         # q[j+1] - q[j]
                    for i in range (N):
                        if dl[i]*dr[i] > 0:
        
                            d[i] = (2*dl[i]*dr[i])/(dl[i]+dr[i])
                    
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
                        F_eta[j] = 0.5 * (-u_L - u_R - eta_R + eta_L)
                        F_u[j]   = 0.5 * (-eta_L - eta_R - u_R + u_L)
                    
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
        
                forcante = self.forcante(eta = cond_eta, u = cond_u)
        
                #primeiro estágio ssprk33
                eta_1 = cond_eta + self.dom.dt * (muscl(cond_eta, cond_u)['new_q_eta'] + forcante)
                u_1 = cond_u + self.dom.dt * muscl(cond_eta,cond_u)['new_q_u']
                
                #segundo estágio ssprk33
                eta_2 = 0.75*cond_eta + 0.25*eta_1 + 0.25*self.dom.dt*muscl(eta_1, u_1)['new_q_eta']
                u_2 = 0.75*cond_u + 0.25*u_1 + 0.25*self.dom.dt*muscl(eta_1, u_1)['new_q_u']      
                
                #terceiro estágio ssprk33
                eta_3 = (1/3)*cond_eta + (2/3)*eta_2 + (2/3)*self.dom.dt*muscl(eta_2, u_2)['new_q_eta'] + self.dom.dt * forcante
                u_3 = (1/3)*cond_u + (2/3)*u_2 + (2/3)*self.dom.dt*muscl(eta_2, u_2)['new_q_u']      
        """

        """
        ###################### versão 23/07/2026 #########################
        ###################função construtor_passos ######################
                else:
            if np.any(np.isclose(janela_de_observacao, self.first_sample)): #x_0 é compatível com a discretização
                x_ultimo = self.first_sample + (self.n_amostras-1)*self.Delta_x
                
                if np.any(np.isclose(janela_de_observacao,x_ultimo)):#x_j é compatível com a discretização
                    pass #os dados informados são compatíves com discretização

                else: #x_0 é compatível com a discretização mas xj não
                    Delta_x_max = (2 - self.first_sample)/self.n_amostras #maior delta_x para a primeira amostra fornecida

                    if Delta_x_max >= self.dom.dx: # ajustando o delta_x para caber as amostras caso x_0 permita
                        self.Delta_x = np.floor(Delta_x_max/self.dom.dx)*self.dom.dx if np.ceil(Delta_x_max/self.dom.dx) != 0 else self.dom.dx

                    else: # ajustando o x0 ao delta_x
                        self.Delta_x = self.dom.dx
                        self.first_sample = 2 - (self.n_amostras+3)*self.dom.dx


            else: #x_0 não está na janela de observação
                self.first_sample = janela_de_observacao[np.argmin(np.abs(janela_de_observacao - self.first_sample))]
                if np.any(np.isclose(janela_de_observacao, self.first_sample)): #x_0 é compatível com a discretização
                    x_ultimo = self.first_sample + (self.n_amostras-1)*self.Delta_x
                    #print(f'ultima amostras = {x_ultimo}')
                if np.any(np.isclose(janela_de_observacao,x_ultimo)):#x_j é compatível com a discretização
                    pass #os dados informados são compatíves com discretização

                else: #x_0 é compatível com a discretização mas xj não
                    Delta_x_max = (2 - self.first_sample)/self.n_amostras #maior delta_x para a primeira amostra fornecida

                    if Delta_x_max >= self.dom.dx: # ajustando o delta_x para caber as amostras caso x_0 permita
                        self.Delta_x = np.floor(Delta_x_max/self.dom.dx)*self.dom.dx if np.ceil(Delta_x_max/self.dom.dx) != 0 else self.dom.dx

                    else: # ajustando o x0 ao delta_x
                        self.Delta_x = self.dom.dx
                        self.first_sample = 2 - (self.n_amostras+3)*self.dom.dx
                #print('ainda incompatível!')




        
        ###################### versão 22/07/2026 ########################
        ###################função construtor_passos ######################
        print('---------------------------')
        print(f'Delta x informado {self.Delta_x}')
        print(f'x0 informado {self.first_sample}')
        print('---------------------------')
        print('')
        
        if (self.Delta_x < 0) or (self.Delta_x>2): #evitar valores de Delta_x impossíveis
            self.Delta_x = 2/self.n_amostras

        if (np.any(np.isclose(self.dom.x, self.first_sample))) and (self.first_sample > 0) and (self.first_sample < 2):#se a primeira amostra estiver tudo ok

            self.first_sample = self.first_sample
            
        if (self.first_sample < 0) or (self.first_sample > 2): # se estiver fora da janela

            self.first_sample  = self.dom.x[np.argmin(np.abs(self.dom.x - 0.09))+1]
            
        if (not np.any(np.isclose(self.dom.x, self.first_sample))): # se estiver na janela mas não é compatível com a discretização
            

            self.first_sample = self.dom.x[np.argmin(np.abs(self.dom.x - self.first_sample))]

        if (self.first_sample == 0):

            self.first_sample = self.dom.x[np.argmin(np.abs(self.dom.x - self.first_sample))+1]
            
        if (self.first_sample == 2):

            self.first_sample = self.dom.x[np.argmin(np.abs(self.dom.x - self.first_sample))-1]
            
        #garantindo que todas as amostra estejam na janela de observação
        
        Delta_x_max = (2 - self.first_sample)/self.n_amostras #maior delta_x para a primeira amostra fornecida

        #1º caso delta x é compativel porém deve ser múltiplo de dx
        if (self.dom.dx<=self.Delta_x) and (self.Delta_x <= Delta_x_max):
            self.Delta_x = math.floor(self.Delta_x/self.dom.dx)*self.dom.dx if math.floor(self.Delta_x/self.dom.dx) != 0 else self.dom.dx

        #2º caso delta x é incompatível compativel pois é menor que o mínimo necessário
        if (self.Delta_x <= self.dom.dx) and (self.dom.dx <= self.Delta_x_max):
            self.Delta_x = self.dom.dx

        #3º caso delta x é incompativel pois é maior que o máximo possível
        if (self.dom.dx<=Delta_x_max) and (Delta_x_max <= self.Delta_x):
            self.Delta_x = math.floor(Delta_x_max/self.dom.dx)*self.dom.dx

            
            
        #4º caso a primeira amostra está próximo de mais do fim da janela de observação
        if (Delta_x_max < self.dom.dx) or (self.first_sample < 0):
            print('identifocou o if que eu quero')
            
            if self.first_sample <= 0:
                print('identifocou primeira amostra negativa')
                self.Delta_x = math.floor(self.Delta_x/self.dom.dx)*self.dom.dx
                self.first_sample = (2-self.dom.dx)+(self.n_amostras-1) * self.Delta_x
                print(f'primeira amostra = {self.first_sample}')
                print(f'Delta_x  = {self.Delta_x}')


            if   self.first_sample <=0:
                k=0
                while ((self.first_sample <=0) and (k<1000)):
                    self.Delta_x = self.Delta_x-k*self.dom.dx
                    self.first_sample = (2-self.dom.dx)-(self.n_amostras-1) * self.Delta_x
                    k += 1
                if k >= 1000:
                    print('Passos insuficientes')


        else:
            
            self.Delta_x = math.floor(self.Delta_x/self.dom.dx)*self.dom.dx
            self.first_sample = (2-self.dom.dx)-(self.n_amostras-1) * self.Delta_x
            

            if   self.first_sample <=0:
                k=0
                while ((self.first_sample <=0) and (k<1000)):
                    self.Delta_x = self.Delta_x-k*self.dom.dx
                    self.first_sample = (2-self.dom.dx)-(self.n_amostras-1) * self.Delta_x
                    k += 1
                if k >= 1000:
                    print('Passos insuficientes')

        xj = np.array([self.first_sample + i*self.Delta_x for i in range(self.n_amostras)])
        passos = [np.where(self.dom.x == xj[i])[0][0].item() for i in range(self.n_amostras)]


        print('---------------------------')
        print(f'Delta x utilizado {self.Delta_x}')
        print(f'x0 utilizado {self.first_sample}')
        print('---------------------------')
        print('')
        print('---------------------------')
        print(f'Passos adotados {passos}')
        print(f'Vetor x_j {xj}')
        print('---------------------------')




        ######################### Fim da função cosntrutor_passos ###########




                if self.ruido:
            phi_t = self.matriz_com_amostras_ruido()
            primeira_parcela = np.sum(self.matriz_ruido, axis=1) - self.n_amostras * phi_n 
            segunda_parcela = []
            for i, passos in enumerate(self.passos):
                   segunda_parcela = np.roll(phi_t[:, i]-phi_n, 2*passos)
            grad = (-self.n_amostras/4)*(primeira_parcela + segunda_parcela)
        




        print(f'a cfl atua é {cfl}')
        max_0 = np.where(max(sol.solucao_analitica_eta(tempo = 0)))
        max_1 = np.where(max(sol.solucao_analitica_eta(tempo = 1)))
        um_passo = max_0-max_1
        print(f'o máximo em zero passos {max_0}')
        print(f'O máximo em um passo {max_1}')
        print(f'diferenca dos máximos {um_passo}')
        
        
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