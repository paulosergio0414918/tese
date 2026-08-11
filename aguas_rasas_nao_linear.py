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


class SolucaoAguasRasasNaoLinear:
    """ Solucao analitica e numerica da equação de águas rasas."""

    def __init__(self,
                 dom: Dominio,
                 condicao: str = "condicao_paper",
                 H: float = 1
                 ):
        
        self.dom = dom
        self.condicao = condicao
        self.variacao_energia = 0
        self.cfl = self.dom.cfl
        self.H = H

    def eta_zero(self, 
                 x: np.ndarray = None
                 ) -> np.ndarray:
        """Define a condição inicial para variável η"""
        condicao = ci.Funcoes2d()
        if x is None:
            x = self.dom.x
        if self.condicao == "condicao_caixa":
            return condicao.condicao_caixa(x)
        
        else:
            return np.array(condicao.condicao_paper(x))
    
    def h_zero(self, 
                 x: np.ndarray = None
                 ) -> np.ndarray:
        """Define a condição inicial para variável η"""
        condicao = ci.Funcoes2d()
        if x is None:
            x = self.dom.x
        if self.condicao == "condicao_caixa":
            return self.H+(condicao.condicao_caixa(x))
        
        else:
            return self.H+np.array(condicao.condicao_paper(x))

    def u_zero(self,
               x: np.ndarray = None) -> np.ndarray:
        """Define a condição inicial para variável u"""
        if x is None:
            x = self.dom.N
            return np.zeros(x)
        
        elif isinstance(x,np.ndarray):
            return np.zeros(len(x))

    def alpha(self,
                 h: np.ndarray,
                 u: np.ndarray
                 ):
        v_left = np.abs(u) + np.sqrt(np.array(h))
        v_right = np.roll(v_left,-1)
        alpha_j = np.maximum(v_left,v_right)
        
        return {
            "alpha_j" : alpha_j,
            "alpha_max": np.max(alpha_j)
        }

    def calculo_delta_t(self, 
                    h: np.ndarray,
                    u: np.ndarray
                    ):
        return self.cfl*self.dom.dx/self.alpha(h,u)["alpha_max"]

    def rusanov_euler(
            self,
            h: np.ndarray = None,
            u: np.ndarray = None,
            H: float = 1
            ) -> np.ndarray:
        #no caso não linear é prociso também contruir o vetor t
        
        h_bar = np.array(h)
        u_bar = np.array(u)
        prod = h_bar*u_bar
        sum = 0.5 * u_bar ** 2 + h_bar
        dt = self.calculo_delta_t(h_bar, u_bar)
        dx = self.dom.dx
        alpha = self.alpha(h_bar,u_bar)["alpha_max"]
        f_h = 0.5*(prod + np.roll(prod,-1) - alpha*(np.roll(h_bar,-1)- h_bar)) 
        f_u  = 0.5*(sum + np.roll(sum,-1)-alpha*(np.roll(u_bar,-1)-u_bar))
        
        h_final = h_bar - dt/dx*(f_h - np.roll(f_h, 1))
        u_final = u_bar - dt/dx * (f_u - np.roll(f_u, 1))
        return{
            'h_final': h_final,
            'u_final': u_final,
            'dt': dt
        }

    def malha_c(
                self,
                eta: np.ndarray = None,
                u: np.ndarray = None
                ) -> np.ndarray:
            #no caso não linear é prociso também contruir o vetor t
            
            eta_bar = np.array(eta)
            u_bar = np.array(u)
            dt = self.dom.cfl*self.dom.dx/np.max(np.abs(u_bar) + np.sqrt(self.H + eta_bar))

            def right_side(eta_bar, u_bar): #obtain the right sido of equation           

                def u_centro(vector): # put the vector u in centre cell
                    return (np.roll(vector,1) + vector)/2

                def eta_interface(vector): # put the eta vector in interface cell
                    return (np.roll(vector,-1) + vector)/2

                def diff_eta(vector): # calculate the finite diference 
                    return  (np.roll(vector,1) - vector)/self.dom.dx

                def diff_u(vector): # calculate the finite diference 
                    return  (vector - np.roll(vector,-1) )/self.dom.dx


                eta_n = diff_eta((self.H + eta_interface(eta_bar))*u_bar)
                u_n = diff_u((0.5*u_centro(u_bar)*u_centro(u)+eta_bar))

                return{
                'eta_final': eta_n,
                'u_final': u_n
                 }

            
            #primeiro estágio ssprk33
            eta_1 = eta_bar + dt * right_side(eta_bar, u)['eta_final']
            u_1 = u_bar + dt * right_side(eta_bar, u)['u_final']
            
            #segundo estágio ssprk33
            eta_2 = 0.75*eta_bar + 0.25*eta_1 + 0.25*dt*right_side(eta_1, u_1)['eta_final']
            u_2 = 0.75*u_bar + 0.25*u_1 + 0.25*dt*right_side(eta_1, u_1)['u_final']      
            
            #terceiro estágio ssprk33
            eta_3 = (1/3)*eta_bar + (2/3)*eta_2 + (2/3)*dt*right_side(eta_2, u_2)['eta_final']
            u_3 = (1/3)*u_bar + (2/3)*u_2 + (2/3)*dt*right_side(eta_2, u_2)['u_final']       
                        

            return{
                'eta_final': eta_3,
                'u_final': u_3,
                'dt': dt
            }

    def muscl_ssprk(self,
                cond_h: np.ndarray = None,
                cond_u: np.ndarray = None,
                modo: str = "muscl_ssprk33"
                ):
        """Avança uma unidade de tempo usando SSPRK22 """
        #construindo as inclinações delta_j
        h_bar = np.array(cond_h)
        u_bar = np.array(cond_u)
        dt = self.calculo_delta_t(h_bar, u_bar)
        
        def delta_minmod(q):
            N = len(q)
            d = np.zeros(N)
            # diferenças com periodicidade
            dl = q - np.roll(q, 1)          # q[j] - q[j-1]
            dr = np.roll(q, -1) - q         # q[j+1] - q[j]
            mask = (dl * dr) > 0
            d[mask] = np.where(np.abs(dl) < np.abs(dr), dl, dr)[mask]
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
        def compute_fluxes(h, u):
            N = len(h)
            d_h= delta(h)
            d_u = delta(u)
            
            F_h = np.zeros(N)
            F_u   = np.zeros(N)
            
            for j in range(N):
                jp1 = (j + 1) % N   # periodicidade
                
                # Reconstrução MUSCL na interface j (entre j e j+1)
                h_L = h[j] + 0.5 * d_h[j]
                u_L   = u[j]   + 0.5 * d_u[j]
                h_R = h[jp1] - 0.5 * d_h[jp1]
                u_R   = u[jp1] - 0.5 * d_u[jp1]
                
                # Fluxo de Godunov para o sistema linearizado (|A|=I)
                F_h[j] = 0.5 * (u_L + u_R - (h_R - h_L))
                F_u[j] = 0.5 * (h_L + h_R - (u_R - u_L))
            
            return F_h, F_u
        
        def muscl(q_h, q_u):
            F_h, F_u = compute_fluxes(q_h, q_u)
            dx = self.dom.dx
            # dq/dt = - (F_j - F_{j-1}) / dx
            new_q_h = -(F_h - np.roll(F_h, 1)) / dx
            new_q_u   = -(F_u   - np.roll(F_u, 1))   / dx
    
            return {
                'new_q_h' : new_q_h  ,
                'new_q_u' : new_q_u
            } 

        if  modo == "muscl_ssprk22":
            #primeiro estágio ssprk22
            h_1 = h_bar + dt * muscl(h_bar, u_bar)['new_q_h']
            u_1 = u_bar + dt * muscl(h_bar,u_bar)['new_q_u']
            
            #segundo estágio ssprk22
            h_2 = 0.5*h_bar + 0.5*h_1 + 0.5*dt*muscl(h_1, u_1)['new_q_h']
            u_2 = 0.5*u_bar + 0.5*u_1 + 0.5*dt*muscl(h_1, u_1)['new_q_u']      
            
            return {
                    'h_final' : h_2,
                    'u_final': u_2,
                    'eta_final': h_2-self.H,
                    'dt': dt
                }     
    
        elif  modo == "muscl_ssprk33":
            #primeiro estágio ssprk33
            h_1 = h_bar + dt * muscl(h_bar, u_bar)['new_q_h']
            u_1 = u_bar + dt * muscl(h_bar,u_bar)['new_q_u']
            
            #segundo estágio ssprk33
            h_2 = 0.75*h_bar + 0.25*h_1 + 0.25*dt*muscl(h_1, u_1)['new_q_h']
            u_2 = 0.75*u_bar + 0.25*u_1 + 0.25*dt*muscl(h_1, u_1)['new_q_u']      
            
            #terceiro estágio ssprk33
            h_3 = (1/3)*h_bar + (2/3)*h_2 + (2/3)*dt*muscl(h_2, u_2)['new_q_h']
            u_3 = (1/3)*u_bar + (2/3)*u_2 + (2/3)*dt*muscl(h_2, u_2)['new_q_u']      
            
            return {
                    'h_final' : h_3,
                    'u_final': u_3,
                    'eta_final': h_3 - self.H,
                    'dt': dt
                }

    def solucao_numerica(self,
                         solucao_eta: np.ndarray = None, # condição incial para eta
                         solucao_u: np.ndarray = None, #condição inicial para u
                         solucao_h:  np.ndarray = None, # condição incial para h
                         tempo: int = None, # tempo de execução do método
                         modo: str = "malha_c" # modelo de execução
                         ):
        """Calcula a solução de águas rasas após vários instantes."""


            
        if solucao_h is None:
            solucao_h = self.h_zero()

        else:
            if modo == "malha_c":
                print('Cuidado! Para malha c é necessário uma condição para η e para h.')
    
        if solucao_u is None:
            solucao_u = self.u_zero()

        if solucao_eta is None:
            solucao_eta = self.eta_zero()

        if modo == "rusanov_euler":

            if tempo is None:
                tempo = self.dom.M

                propagacao = self.rusanov_euler(eta = solucao_h, u= solucao_u)
                flag = 0
                time = 0
                while True:
                    flag += 1
                    time += propagacao['dt']
                    h_final = propagacao['h_final']
                    u_final = propagacao['u_final']
                    propagacao = self.rusanov_euler(eta = h_final, u= u_final)

                    if (flag > 1000) or (time > tempo):
                        break
            else:
                propagacao = self.rusanov_euler(eta = solucao_h, u= solucao_u)

                for _ in range(int(tempo)+1):
                    h_final = propagacao['h_final']
                    u_final = propagacao['u_final']
                    propagacao = self.rusanov_euler(eta = h_final, u= u_final)

            return {
                'h' : h_final,
                'u'   : u_final,
                'eta' : h_final-self.H
            }

        elif modo == "muscl_ssprk22":
            
            if tempo is None:
                tempo = self.dom.M

                propagacao = self.muscl_ssprk(solucao_eta,solucao_u, modo = "muscl_ssprk22")
                flag = 0
                time = 0
                while True:
                    flag += 1
                    time += propagacao['dt']
                    h_final = propagacao['h_final']
                    u_final = propagacao['u_final']
                    propagacao = self.muscl_ssprk(h_final,u_final, modo = "muscl_ssprk22")

                    if (flag > 1000) or (time > tempo):
                        break
            else:
                propagacao = self.muscl_ssprk(eta = solucao_h, u= solucao_u, modo = "muscl_ssprk22")

                for _ in range(int(tempo)+1):
                    h_final = propagacao['h_final']
                    u_final = propagacao['u_final']
                    propagacao = self.muscl_ssprk(eta = h_final, u= u_final, modo = "muscl_ssprk22")

            return {
                'h' : h_final,
                'u'   : u_final,
                'eta' : h_final-self.H
                }

        elif modo == "muscl_ssprk33":
        
            if tempo is None:
                tempo = self.dom.M

                propagacao = self.muscl_ssprk(solucao_eta,solucao_u, modo = "muscl_ssprk33")
                flag = 0
                time = 0
                while True:
                    flag += 1
                    time += propagacao['dt']
                    h_final = propagacao['h_final']
                    u_final = propagacao['u_final']
                    propagacao = self.muscl_ssprk(h_final,u_final, modo = "muscl_ssprk33")
                    
                    if (flag > 1000) or (time > tempo):
                        break
            else:
                time = 0
                propagacao = self.muscl_ssprk(cond_h= solucao_h, cond_u = solucao_u, modo = "muscl_ssprk33")

                for j in range(int(tempo)+1):
                    time += propagacao['dt']
                    h_final = propagacao['h_final']
                    u_final = propagacao['u_final']
                    propagacao = self.muscl_ssprk(cond_h = h_final, cond_u= u_final, modo = "muscl_ssprk33")

            return {
                'h' : h_final,
                'u'   : u_final,
                'eta' : h_final-self.H
                }

        elif modo == "malha_c":
            
            
            #print(f'condição incial eta \n{np.max(solucao_eta)}')
            #print(f'teste da raiz \n{np.max(np.max(solucao_eta))}')
            
            if tempo is None:
                tempo = self.dom.M

                propagacao = self.malha_c(eta = solucao_eta, u = solucao_u)
                dt = propagacao['dt']
                flag = 0
                time = 0
                while True:
                    flag += 1
                    time += propagacao['dt']
                    eta_final = propagacao['eta_final']
                    u_final = propagacao['u_final']
                    propagacao = self.malha_c(eta = eta_final, u = u_final)
                    
                    if (flag > 1000) or (time > tempo):
                        break
            else:
                time = 0
                propagacao = self.malha_c(eta= solucao_eta, u = solucao_u)

                for i in range(int(tempo)+1):
                    
                    eta_final = propagacao['eta_final']
                    u_final = propagacao['u_final']
                    #print(f'max da solução {np.max(u_final)}')
                    propagacao = self.malha_c(eta = eta_final , u= u_final,)

                    if i == tempo:
                        time = propagacao['dt']


            return {
                'u'   : u_final,
                'eta' : eta_final,
                'time': time
            }
                    
        else:
            print("Modo não definido")
       
class Validacao(SolucaoAguasRasasNaoLinear):
    """validação do método numérico"""
  
    def __init__(self,
                dom: Dominio,
                testes: int = 6,
                modo: str = "godunov_euler"
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
            tab = Table(title = r"Ordem de convergência para $\eta$ para modelo {}.".format(self.modo))
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
                s = SolucaoAguasRasasNaoLinear(domi)
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
        sol = SolucaoAguasRasasNaoLinear(domi)
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
        sol = SolucaoAguasRasasNaoLinear(domi)
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


#TODO:
#! implementar o método de obtenção de amostras
#! calculo do custo funcional
#! calculo do erro de reconstrução
#! implementar o gradiente
#! implementar o gradiente descedente
#! constuir o gráfico do custo de funcional
#! construir o grafico da reconstrução da condição inicial.
     
class Assimilacao(SolucaoAguasRasasNaoLinear):

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
        self.sol = SolucaoAguasRasasNaoLinear(self.dom)
        self.modo = modo
        self._matriz_com_amostras = None
        self._matriz_com_amostras_ruido = None
        self.vetor_custo = []
        #self.vetor_ruido = [random.uniform(0, 0.005) for i in range(self.dom.N)]
        self.matriz_ruido = np.array([[random.gauss(0, self.standard_deviation) for _ in range(self.dom.M)] for _ in range(self.n_amostras)]) # matriz de ordem n_amostrasxM
        self.E = np.linalg.norm(self.vetor_ruido)/self.n_amostras
        self.tj = [((dom.M*(dom.T-(dom.T/self.n_amostras)*i))/2)*dom.dt for i in range(self.n_amostras)]

        self.matriz_de_amostras_ruido()
    #TODO: Falta testar o construtor de passos
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
        #if self._matriz_com_amostras is None:
        """Gera uma matriz contendo as amostras sem perturbação"""
        matriz = np.zeros((self.n_amostras, self.dom.M)) # as amostras serão armazenadas em linhas 
        steps = self.construtor_passos()['passos']
        
        #FIXME:
        #! temos um problema! Como sincronizar o tempo no caso não linear.
        for i in range(self.n_amostras):
            for j in range(self.dom.M):
                soluction = sol.solucao_numerica(modo='malha_c', tempo = j)
                matriz[i, j] = soluction[steps[i]]
            
        self.matriz_com_amostras = matriz
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
    
    #discretizacao = "rusanov_euler"
    #discretizacao = "muscl_ssprk22"
    discretizacao = "malha_c"
    #dom = Dominio(N=1024, M=256) #cfl = 1
    dom = Dominio(N=1024, M=320) #cfl = 0.8
    #dom = Dominio(N=1024, M=512) #cfl = 0.5
    sol = SolucaoAguasRasasNaoLinear(dom)
    #sol = SolucaoAguasRasasNaoLinear(dom, cfl= cfl)
    cfl = dom.cfl
    val = Validacao(dom, modo = discretizacao)
    
    
    op = 6
    it = 2**8


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
            from aguas_rasas_linear import SolucaoAguasRasas as swel
            #solu = swel.solucao_analitica_eta(dom = dom)
            solucao = sol.solucao_numerica(modo = discretizacao, tempo = i)
            y = solucao['eta']
            #z = solu
            e_22 = np.linalg.norm(y)
                        
            if e_22 > 1:
                print(f"considerando {discretizacao} o erro { e_22}")
                print("erro muito grande.")
                break
            plt.clf()
            plt.xlim(-dom.L, dom.L) # x limit
            plt.plot(dom.x, y, label = 'Solução numérica ' )
            #plt.plot(dom.x, z, label = 'Solução Analítica')
            plt.title(f'Execução {i+1} de {it} do modelo {discretizacao} com cfl = {cfl}.')
            plt.legend()

            #plt.show(block = False)
            plt.pause(0.1)

        plt.show()               

    elif op == 1: # teste da cfl
        print(f'dt = {dom.dt}')
        print(f'dx = {dom.dx}')
        print(f'cfl = {sol.calculo_cfl()}')

    elif op == 0: # teste da solução analítica para η
        y = sol.solucao_analitica_eta()

        graf = cdg.Grafico2d(dom.x,y)
        graf.plot2d()

