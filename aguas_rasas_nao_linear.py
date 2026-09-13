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

def _worker_assimilacao_nao_linear(args):
    import dominio as _dom_mod
    import traceback
    n_amostras, N, M, ruido, first_sample, Delta_x, iteracoes = args
    try:
        dom_w = _dom_mod.Dominio(N=N, M=M)
        ass_w = Assimilacao(dom_w, modo="malha_c", n_amostras=n_amostras,
                            ruido=ruido, first_sample=first_sample,
                            Delta_x=Delta_x)
        passos = ass_w.construtor_passos()['passos']
        xj = ass_w.construtor_passos()['xj']
        print(f"[start] n={n_amostras} "
              f"fs={ass_w.first_sample:.4f} dx={ass_w.Delta_x:.4f} "
              f"passos={passos} xj={xj} T={ass_w.T:.4f}")

        gd_ot = ass_w.gradiente_descendente_otimizado(it=iteracoes)
        gd_no = ass_w.gradiente_descendente(it=iteracoes)
        print(f"[done]  n={n_amostras} "
              f"err_ot[0]={gd_ot['error'][0]:.3e} "
              f"err_ot[-1]={gd_ot['error'][-1]:.3e}")
        return {'n': n_amostras, 'ot': gd_ot, 'no': gd_no}
    except Exception as e:
        print(f"[FAIL] n={n_amostras}: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise


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
                u: np.ndarray = None,
                dt: float = None
                ) -> np.ndarray:
            #no caso não linear é prociso também contruir o vetor t



            eta_bar = np.array(eta)
            u_bar = np.array(u)
            if dt is None:
                dt = self.dom.cfl * self.dom.dx / np.max(np.abs(u_bar) + np.sqrt(self.H + eta_bar))

            def right_side(eta_bar, u_bar): #obtain the right side of equation           

                def u_centro(vector): # put the vector u in centre cell
                    return (np.roll(vector,1) + vector)/2

                def eta_interface(vector): # put the eta vector in interface cell
                    return (np.roll(vector,-1) + vector)/2

                def diff_eta(vector): # calculate the finite diference 
                    return  (np.roll(vector,1) - vector)/self.dom.dx

                def diff_u(vector): # calculate the finite diference 
                    return  (vector - np.roll(vector,-1) )/self.dom.dx


                eta_n = diff_eta((self.H + eta_interface(eta_bar))*u_bar)
                #u_n = diff_u((0.5*u_centro(u_bar)*u_centro(u)+eta_bar)) <- antigo
                u_n = diff_u((0.5*u_centro(u_bar)*u_centro(u_bar)+eta_bar)) 

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
                    
                    if (flag > 2000) or (time > tempo):
                        break
            else:
                time = 0
                #propagacao = self.malha_c(eta= solucao_eta, u = solucao_u)
                eta_final = solucao_eta
                u_final = solucao_u
                for i in range(int(tempo)):
                    propagacao = self.malha_c(eta = eta_final , u= u_final,)
                    eta_final = propagacao['eta_final']
                    u_final = propagacao['u_final']
                    #print(f'max da solução {np.max(u_final)}')

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

'''class old_Assimilacao(SolucaoAguasRasasNaoLinear):

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
'''

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
        self.sol = SolucaoAguasRasasNaoLinear(self.dom, condicao=self.condicao)
        self.modo = modo
        self.matriz_com_amostras = None
        self.matriz_com_amostras_ruido = None
        self.v_t_ref = None                     
        self.T = None  
        self.vetor_custo = []
        self.matriz_ruido = np.array([[random.gauss(0, self.standard_deviation) for _ in range(self.dom.M)] for _ in range(self.n_amostras)]) # ( n_amostras, M)
        self.E = np.abs(np.mean(np.sum(self.matriz_ruido, axis=1)))
        self.valida_passos()
        #self.matriz_de_amostras()
        self.constroi_verdade()  
        self._print_passos_done = True
                
    def valida_passos(self,print_info: bool = False): # metodo para evitar que os passos de assimilação não pertençam ao domínio espacial do problema      
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

    def passos_efetivos(self):# metodo para evitar sobreescrever os passos de assimilação no meio do caminho
        xj = np.array([self.first_sample + i*self.Delta_x for i in range(self.n_amostras)])
        position = [int(np.where(np.isclose(self.dom.x, xj[i]))[0][0]) for i in range(self.n_amostras)]
        return {'passos': position, 'xj': xj}

    def construtor_passos(self, print_info=False): # metodo para evitar sobreescrever os passos de assimilação no meio do caminho
        if print_info: 
            print(f"passos de assimilação {self.passos_efetivos()['passos']}") 

        return self.passos_efetivos()
    
    def constroi_verdade(self): # (evitar que a matriz de amostras seja sobre escrita durante o processo de assimilação 
        """Roda o modelo com φ_true, guarda η em x_j nos instantes k=0..M-1
        e o vetor de tempos físicos acumulados v_t_ref."""

        steps = self.construtor_passos()['passos'] #gera os passos de assimilação
        M     = self.dom.M # discretização temporal 

        eta_traj = np.zeros((self.dom.N, M)) #matriz para armazenar as trajetórias 
        v_t      = np.zeros(M) # vetor para armazenar os passos de tempo

        e = self.sol.eta_zero() # condição inicial verdadeira para eta
        u = self.sol.u_zero() # condição inicial verdadeira para u
        eta_traj[:, 0] = e # popula a primeira coluna da matriz da trajetória
        v_t[0] = 0.0 # popula a primeira coluna da matriz da trajetória
        t = 0.0 # acumula o tempo de execução

        for k in range(1, M):
            out = self.sol.malha_c(eta=e, u=u)  # gera as soluções usando malha c e a condição inicial verdadeira
            e, u = out['eta_final'], out['u_final'] #
            t += out['dt']
            eta_traj[:, k] = e # popula as demais colunas da matriz 
            v_t[k] = t # popula os demais passos de tempo

        self.matriz_com_amostras = eta_traj[steps, :] # carrega apenas os passos x_j na matriz de amostras
        self.v_t_ref             = v_t # gera o vetor com os passos de tempo que serão usados no processo de assimilação
        self.T                   = v_t[-1] # apresenta o tempo final de assimilação que deve ser próximo de 2 com margem de erro pois dt é variável

        self.matriz_com_amostras.flags.writeable = False # mantem a matriz de amostras fixo durante rodo o processo de assimilação
        self.v_t_ref.flags.writeable             = False# mantem a matriz de amostras fixo durante rodo o processo de assimilação 

    def dt_sincronizado(self, eta, u, t_atual, t_proximo): # sincroniza o tempo utilizado na discretização do gradiente com tempo do sistema direto
        """Menor entre o dt natural e o tempo que falta até t_proximo."""
        h = self.sol.H + eta
        dt_natural  = self.sol.calculo_delta_t(h, u)
        dt_restante = t_proximo - t_atual
        return min(dt_natural, dt_restante)

    def trajetoria_forecast(self, eta): # método criado para exitar excessivas chamada da solução o que torna a execução lenta.
        """Avança o forecast de 0 até self.T, aterrissando em cada v_t_ref[k].
        Devolve eta_f_obs com shape (n_amostras, M)."""
        steps = self.construtor_passos()['passos']
        M = self.v_t_ref.size
        N = self.dom.N
        eta_f_obs = np.zeros((self.n_amostras, M))
        eta_traj   = np.zeros((N, M));  
        u_traj     = np.zeros((N, M))


        e = np.array(eta)
        u = np.zeros(N)
        t = 0.0
        eta_f_obs[:, 0] = e[steps]
        eta_traj[:, 0]  = e
        u_traj[:, 0] = u

        for k in range(1, M):
            t_prox = self.v_t_ref[k]
            while t < t_prox - 1e-14:
                dt = self.dt_sincronizado(e, u, t, t_prox)
                out = self.sol.malha_c(eta=e, u=u, dt=dt)
                e, u = out['eta_final'], out['u_final']
                t += dt
            eta_f_obs[:, k] = e[steps]
            eta_traj[:, k]  = e
            u_traj[:, k]    = u

        return (eta_f_obs, eta_traj, u_traj)

    #@staticmethod
    def pesos_trapezio(self, t): # método criado para o calculo da regra do trapésio do custo
        w = np.zeros_like(t)
        w[0]    = (t[1]  - t[0])  / 2
        w[-1]   = (t[-1] - t[-2]) / 2
        w[1:-1] = (t[2:] - t[:-2]) / 2
        return w

    def custo_assimilacao(self, eta=None):
        """Funcional de custo de assimilação (caso não linear).

        J[φ] = (1/2) ∫_0^T Σ_j [ η^(f)(x_j, t) - y_j(t) ]² dt

        Usa trajetoria_forecast para obter η^(f)(x_j, t_k) em uma única
        varredura sincronizada com v_t_ref, e integra no tempo com a regra
        do trapézio em grade não uniforme (pesos_trapezio).
        """
        if eta is None:
            eta = self.sol.eta_zero()

        # ---------- observações y_j (n_amostras × M) --------------------
        if self.ruido:
            if self.matriz_com_amostras_ruido is None:
                self.matriz_de_amostras_ruido()
            y_j = self.matriz_com_amostras_ruido
        else:
            if self.matriz_com_amostras is None:
                self.matriz_de_amostras()
            y_j = self.matriz_com_amostras

        # ---------- forecast nos pontos de observação -------------------
        # eta_f_obs[j, k] = η^(f)(x_j, t_k),  t_k = v_t_ref[k]
        eta_f_obs = self.trajetoria_forecast(eta)[0]     # (n_amostras, M)

        # ---------- diferenças quadradas somadas nas amostras -----------
        diff = (eta_f_obs - y_j) ** 2                    # (n_amostras, M)
        sum_diff = np.sum(diff, axis=0)                  # (M,)

        # ---------- integração no tempo (trapézio, grade não uniforme) --
        w = self.pesos_trapezio(self.v_t_ref)
        integral = np.sum(w * sum_diff)

        return 0.5 * integral    

    def old_matriz_de_amostras(self):

        """Gera uma matriz contendo as amostras sem perturbação"""
        
        matriz = np.zeros((self.n_amostras, self.dom.M)) # as amostras serão armazenadas em linhas 

        steps = self.construtor_passos()['passos']


        for i in range(self.n_amostras):
            for j in range(self.dom.M):
                solution = self.sol.solucao_analitica_eta(tempo=j)   
                matriz[i, j] = solution[steps[i]]
            
        self.matriz_com_amostras = matriz
        return matriz # retorna uma matriz de ordem n_amostrasxM

    def matriz_de_amostras(self): #gera a matriz de dados assimilados sem o problema de sobre escrever quando apresentar o gráfico
        """Retorna a matriz de amostras da verdade (já construída em __init__)."""
        if self.matriz_com_amostras is None:
            self.constroi_verdade()
        return self.matriz_com_amostras  
        
    def matriz_de_amostras_ruido(self): #gera a matriz de dados assimilados com ruido
        """Gera uma matriz contendo as amostras com perturbação """
        if self.matriz_com_amostras_ruido is None:
            self.matriz_de_amostras()     
        matriz_com_ruido = self.matriz_com_amostras.copy() # gera uma cópia da matriz de amostras para não precisar construir outra   
        amostras_com_ruido = matriz_com_ruido + self.matriz_ruido
        self.matriz_com_amostras_ruido = amostras_com_ruido
        self.matriz_com_amostras_ruido.flags.writeable = False # garantir que as amostras não seja sobreescritas
        return amostras_com_ruido # retorna uma matriz de ordem n_amostrasxM    

    def forcante(self, u=None, eta=None): # constroi a forçante presente no sistema adjunto separadamente para ecomonizar tempo de execução
        """Forçante do sistema adjunto não linear.

        S_i(t_k) = r_j(t_k)  para  i = step_j  (a célula que contém x_j),
                0          caso contrário,

        com r_j(t_k) = η^(f)(x_j, t_k) - y_j(t_k).

        Uma única varredura temporal via _trajetoria_forecast (que já
        sincroniza com v_t_ref), seguida de espalhamento vetorizado.
        """
        if eta is None:
            eta = self.sol.eta_zero()
        if u is None:
            u = self.sol.u_zero()

        # garante verdade/observações construídas
        if self.matriz_com_amostras is None:
            self.matriz_de_amostras()

        y = (self.matriz_com_amostras_ruido if self.ruido
            else self.matriz_com_amostras)          # (n_amostras, M)

        # ---- UMA varredura temporal do forecast economisa um dia de execução -------------------------
        # eta_f_obs[j, k] = η^(f)(x_j, t_k), com t_k = v_t_ref[k]
        eta_f_obs = self.trajetoria_forecast(eta)[0]   # (n_amostras, M)

        # ---- resíduo nos pontos de observação ---------------------------
        r = eta_f_obs - y                            # (n_amostras, M)

        # ---- espalha nas células que contêm observações -----------------
        steps = self.construtor_passos()['passos']   # lista de índices i
        forcante = np.zeros((self.dom.N, self.dom.M))
        forcante[steps, :] = r                        # popula a matriz da forçante

        return forcante

    def adjoint_rhs(self, eta, u, lam_eta, lam_u):
        """RHS do adjunto em τ = T - t.

        η*_τ = u η*_x + u*_x + S
        u*_τ = (1+η) η*_x + u u*_x
        """
        dx = self.dom.dx
        h  = self.sol.H

        def u_centro(v):  return (np.roll(v, 1) + v) / 2
        def eta_interface(v):  return (v + np.roll(v, -1)) / 2
        def diff_eta(v): return (np.roll(v, 1) - v) / dx
        def diff_u(v): return (v - np.roll(v, -1)) / dx

        # η*_τ  (com sinal do lado direito já incorporado)
        d_eta = (- diff_eta(u * eta_interface(lam_eta))
                + lam_eta * diff_eta(u)
                - diff_eta(lam_u))

        # u*_τ
        d_u = (-eta_interface(h + eta) * diff_u(lam_eta)
            + u * (np.roll(lam_u, -1) - np.roll(lam_u, 1)) / (2 * dx))

        return d_eta, d_u

    def adjoint_step(self, eta_f, u_f, lam_eta, lam_u, dt, S_k):
        """Um passo SSPRK33 do adjunto, avançando em τ."""
        # estágio 1
        de1, du1 = self.adjoint_rhs(eta_f, u_f, lam_eta, lam_u)
        de1 = de1 + S_k
        e1 = lam_eta + dt * de1
        u1 = lam_u   + dt * du1

        # estágio 2
        de2, du2 = self.adjoint_rhs(eta_f, u_f, e1, u1)
        de2 = de2 + S_k
        e2 = 0.75*lam_eta + 0.25*e1 + 0.25*dt * de2
        u2 = 0.75*lam_u   + 0.25*u1 + 0.25*dt * du2

        # estágio 3
        de3, du3 = self.adjoint_rhs(eta_f, u_f, e2, u2)
        de3 = de3 + S_k
        e3 = (1/3)*lam_eta + (2/3)*e2 + (2/3)*dt * de3
        u3 = (1/3)*lam_u   + (2/3)*u2 + (2/3)*dt * du3

        return e3, u3

    def grad(self, cond_eta=None, cond_u=None):
        if cond_eta is None:
            cond_eta = self.sol.eta_zero()

        _ , eta_traj, u_traj = self.trajetoria_forecast(cond_eta) # o _ indica que não utilizarei a primira entrada
        fonte = self.forcante(eta=cond_eta)

        M = self.v_t_ref.size
        w = self.pesos_trapezio(self.v_t_ref)

        lam_eta = np.zeros(self.dom.N)
        lam_u   = np.zeros(self.dom.N)

        for k in range(M - 1, 0, -1):
            dt  = self.v_t_ref[k] - self.v_t_ref[k-1]
            S_k = w[k] * fonte[:, k] / self.dom.dx

            lam_eta, lam_u = self.adjoint_step(
                eta_traj[:, k-1], u_traj[:, k-1],
                lam_eta, lam_u, dt, S_k,
            )

        # injeção final em t = 0
        lam_eta = lam_eta + w[0] * fonte[:, 0] / self.dom.dx

        return {'eta_grad': lam_eta, 'u_grad': lam_u}

    def gradiente_descendente(self, it=10, alpha=0.1):
        """Gradiente descendente com passo fixo alpha (caso não linear).

        Otimiza apenas φ = η(x, 0). A condição inicial de u é dado do
        problema (u(x, 0) = 0) e permanece fixa.
        """
        from tqdm import tqdm

        def reconstruction_error(vet):
            return (np.linalg.norm(vet - self.sol.eta_zero())
                    / np.linalg.norm(self.sol.eta_zero()))

        solucao_final_eta = np.zeros(self.dom.N)        # chute inicial
        error = []
        custo = []
        all_solutions_eta = np.zeros((self.dom.N, it))

        for i in tqdm(range(it), desc="gradiente descendente"):
            grad = self.grad(cond_eta=solucao_final_eta)
            solucao_final_eta = solucao_final_eta - alpha * grad['eta_grad']

            error.append(reconstruction_error(solucao_final_eta))
            custo.append(self.custo_assimilacao(solucao_final_eta))
            all_solutions_eta[:, i] = solucao_final_eta

        return {
            'eta_final': solucao_final_eta,
            'error': error,
            'custo': custo,
            'all_solutions': all_solutions_eta,
        }

    def passo_robusto(self, eta, grad_eta_local, J_atual,
                   alpha_prev=0.1, alpha_min=1e-6, alpha_max=1.0,
                   grad_fun=None):
        """Escolhe alpha com Wolfe → backtracking Armijo → alpha_min."""
        from scipy.optimize import line_search

        pk = -grad_eta_local
        gTp = float(np.dot(grad_eta_local, pk))       # = -||g||²

        if (not np.isfinite(gTp)) or gTp >= 0:
            return alpha_prev, J_atual

        if grad_fun is None:
            grad_fun = lambda e: self.grad(cond_eta=e)['eta_grad']

        try:
            otimi = line_search(
                self.custo_assimilacao, grad_fun, eta, pk,
                c1=1e-4, c2=0.5, maxiter=30,
                old_fval=J_atual,
            )
            alpha = otimi[0]
        except Exception:
            alpha = None

        if alpha is not None and np.isfinite(alpha) and alpha_min <= alpha <= alpha_max:
            Jn = self.custo_assimilacao(eta + alpha * pk)
            if np.isfinite(Jn):
                return float(alpha), float(Jn)

        # backtracking Armijo
        alpha = alpha_prev if alpha_min <= alpha_prev <= alpha_max else 0.1
        for _ in range(40):
            Jn = self.custo_assimilacao(eta + alpha * pk)
            if np.isfinite(Jn) and Jn <= J_atual + 1e-4 * alpha * gTp:
                return float(alpha), float(Jn)
            alpha *= 0.5
            if alpha < alpha_min:
                break

        return alpha_min, self.custo_assimilacao(eta + alpha_min * pk)

    def gradiente_descendente_otimizado(self, it=10):
        """Gradiente descendente com passo robusto. Otimiza só φ = η(x,0)."""
        from tqdm import tqdm

        def reconstruction_error(vet):
            return (np.linalg.norm(vet - self.sol.eta_zero())
                    / np.linalg.norm(self.sol.eta_zero()))

        solucao_final_eta = np.zeros(self.dom.N)
        all_solutions_eta = np.zeros((self.dom.N, it))
        error, custo, alpha = [], [], []

        J_atual = self.custo_assimilacao(solucao_final_eta)

        for i in tqdm(range(it), desc="GD otimizado"):
            g = self.grad(cond_eta=solucao_final_eta)['eta_grad']
            alpha_i, J_atual = self.passo_robusto(
                solucao_final_eta, g, J_atual,
                alpha_prev=alpha[-1] if alpha else 0.1,
            )
            solucao_final_eta = solucao_final_eta - alpha_i * g

            error.append(reconstruction_error(solucao_final_eta))
            custo.append(J_atual)
            alpha.append(alpha_i)
            all_solutions_eta[:, i] = solucao_final_eta

        return {
            'eta_final':     solucao_final_eta,
            'error':         error,
            'custo':         custo,
            'alpha':         alpha,
            'all_solutions': all_solutions_eta,
        }



if __name__ == "__main__":
    import dominio
    import matplotlib.pyplot as plt
    import construtor_de_graficos as cdg
    import numpy as np
    
    #discretizacao = "rusanov_euler"
    #discretizacao = "muscl_ssprk22"
    discretizacao = "malha_c"
    #N=1024; M=256 #cfl = 1
    N=1024; M=320 #cfl = 0.8
    #N=1024; M=512 #cfl = 0.5
    dom = Dominio(N=N, M=M) 
    
    sol = SolucaoAguasRasasNaoLinear(dom)
    #sol = SolucaoAguasRasasNaoLinear(dom, cfl= cfl)
    amos = 2
    ruido = False
    first_sample = .2 # paper uses first_sample = 0.2
    Delta_x =  0.09 # paper uses Delta_x = 0.09 end Delta_x = 0.375 for counter-example
    modo = "malha_c"

    def _tag(v): return f"{v:g}".replace("-", "m").replace(".", "p")

    cfl = dom.cfl
    val = Validacao(dom, modo = discretizacao)
    ass = Assimilacao(dom, modo= modo, n_amostras = amos,
                    ruido=ruido,
                    first_sample = first_sample, 
                    Delta_x =  Delta_x ) 


                     
    op = 18
    it = 2**6
    iteracoes = it



    if op == 18:  # roda 2..6 amostras em paralelo, salva incrementalmente
        import os
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(v, "1")

        from multiprocessing import Pool

        tarefas = [(n, N, M, ruido, first_sample, Delta_x, iteracoes)
                for n in (2, 3, 4, 5, 6)]

        flag2 = 'with_noise' if ruido else 'without_noise'
        sufixo = (f"malha_c_fs_{_tag(first_sample)}"
                f"_deltax_{_tag(Delta_x)}_it_{iteracoes}_{flag2}")

        with Pool(processes=len(tarefas)) as pool:
            for res in pool.imap_unordered(_worker_assimilacao_nao_linear, tarefas):
                n_amostras = res['n']
                gd_ot, gd_no = res['ot'], res['no']
                base = f"data/nao_linear_gd_n{n_amostras}"

                np.savetxt(f"{base}_ot_custo_{sufixo}.csv", np.array(gd_ot['custo']),  delimiter=",")
                np.savetxt(f"{base}_ot_erro_{sufixo}.csv", np.array(gd_ot['error']),  delimiter=",")
                np.savetxt(f"{base}_ot_alpha_{sufixo}.csv",np.array(gd_ot['alpha']),  delimiter=",")
                np.savetxt(f"{base}_no_custo_{sufixo}.csv",np.array(gd_no['custo']),  delimiter=",")
                np.savetxt(f"{base}_no_erro_{sufixo}.csv",np.array(gd_no['error']),  delimiter=",")
                np.savetxt(f"{base}_ot_all_solutions_{sufixo}.csv", gd_ot['all_solutions'].T,  delimiter=",")
                np.savetxt(f"{base}_no_all_solutions_{sufixo}.csv", gd_no['all_solutions'].T,  delimiter=",")

                print(f"[ok] n={n_amostras} salvo.")


    elif op == 9:
        import dominio as _dom_mod
        from types import MethodType

        dom_test = _dom_mod.Dominio(N=64, M=16)
        ass_test = Assimilacao(dom_test, modo="malha_c", n_amostras=2,
                            ruido=False, first_sample=0.2, Delta_x=0.09)

        # ---- MONKEY PATCH: constroi_verdade com dt fixo ----
        def _constroi_verdade_fixo(self):
            steps = self.construtor_passos()['passos']
            M = self.dom.M
            dt_base = self.sol.cfl * self.dom.dx / np.sqrt(self.sol.H)

            eta_traj = np.zeros((self.dom.N, M))
            v_t = np.zeros(M)
            e = self.sol.eta_zero(); u = self.sol.u_zero()
            eta_traj[:, 0] = e

            for k in range(1, M):
                out = self.sol.malha_c(eta=e, u=u, dt=dt_base)
                e, u = out['eta_final'], out['u_final']
                eta_traj[:, k] = e
                v_t[k] = k * dt_base

            self.matriz_com_amostras = eta_traj[steps, :]
            self.v_t_ref = v_t
            self.T = v_t[-1]
            self.matriz_com_amostras.flags.writeable = False
            self.v_t_ref.flags.writeable = False

        ass_test.constroi_verdade = MethodType(_constroi_verdade_fixo, ass_test)
        ass_test.constroi_verdade()

        phi = 0.3 * ass_test.sol.eta_zero()
        i_test = 30
        e_i = np.zeros(dom_test.N); e_i[i_test] = 1.0
        eps = 1e-6

        g_fd  = (ass_test.custo_assimilacao(phi + eps*e_i)
                - ass_test.custo_assimilacao(phi - eps*e_i)) / (2*eps)
        g_adj = ass_test.grad(cond_eta=phi)['eta_grad'][i_test]

        print(f"dt fixo  g_fd={g_fd:+.6e}  g_adj={g_adj:+.6e}  razão={g_fd/g_adj:+.4f}")

    elif op == 8:  # validação da forçante (caso não linear)
        import numpy as np

        # --- setup mínimo ---------------------------------------------------
        ass = Assimilacao(dom, modo="malha_c", n_amostras=amos,
                          ruido=ruido,
                          first_sample=first_sample, Delta_x=Delta_x)

        eta0 = np.zeros(dom.N)          # φ^f = 0 (chute inicial padrão)
        u0   = np.zeros(dom.N)
        fonte = ass.forcante(eta=eta0, u=u0)

        # --- (a) suporte: só as células de observação têm fonte não nula ----
        steps = ass.construtor_passos()['passos']
        linhas_nao_nulas = np.where(np.any(fonte != 0, axis=1))[0]

        print("(a) células com fonte :", linhas_nao_nulas)
        print("    esperado          :", steps)
        assert set(linhas_nao_nulas.tolist()) == set(steps), \
            "forçante está fora das células de observação"

        # --- (b) valor: fonte[steps, :] == resíduo η^f - y ------------------
        eta_f_obs = ass._trajetoria_forecast(eta0)          # (n_amostras, M)
        y         = (ass.matriz_com_amostras_ruido if ass.ruido
                     else ass.matriz_com_amostras)
        r_esperado = eta_f_obs - y                           # (n_amostras, M)

        ok = np.allclose(fonte[steps, :], r_esperado)
        print("(b) fonte[steps,:] == η^f - y :", ok)
        if not ok:
            erro = np.max(np.abs(fonte[steps, :] - r_esperado))
            print(f"    erro máximo = {erro:.3e}")
        assert ok, "resíduo não está sendo propagado corretamente"

        # --- (c) zeros fora das observações --------------------------------
        mascara = np.ones(dom.N, dtype=bool)
        mascara[steps] = False
        zerado = np.all(fonte[mascara, :] == 0)
        print("(c) fora das observações é zero :", zerado)
        assert zerado, "há vazamento de fonte fora das células de observação"

        print("\nTodos os testes passaram.")

    elif op == 7: # teste da 
        # 1) Fixe uma condição inicial
        eta0 = np.zeros(dom.N) + 0.001 * np.random.randn(dom.N)

        # 2) Calcule a forçante
        fonte = ass.forcante(eta=eta0)

        # 3) Verifique que só as células com observações são não nulas
        steps = ass.construtor_passos()['passos']
        linhas_nao_nulas = np.where(np.any(fonte != 0, axis=1))[0]
        print("células com fonte:", linhas_nao_nulas)
        print("esperado         :", steps)
        assert set(linhas_nao_nulas) == set(steps)

    elif op == 6: # teste da ordem de convergênia da solução numérica
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

