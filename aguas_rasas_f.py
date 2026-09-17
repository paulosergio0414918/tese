#shallow water in f 
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

class SolucaoAguasRasasF:
    """ Solucao analitica e numerica da equação de águas rasas."""

    def __init__(self,
                 dom: Dominio,
                 cond: str = "condicao_paper",
                 H: float = 1,
                 f: float = 7.292e-05,
                 g: float = 9.81
                 ):
        
        self.dom = dom
        self.cond = cond
        self.H = H
        self.f = f
        self.g = g
    
    def h_zero(self, 
                 x: np.ndarray = None) -> np.ndarray:
        """Define a condição inicial para variável η"""
        condicao = ci.Funcoes2d()
        if x is None:
            x = self.dom.x
        if self.cond == "condicao_caixa":
            return condicao.condicao_caixa(x) + self.H
        
        else:
            return condicao.condicao_paper(x) + self.H

    def u_zero(self,
               x: np.ndarray = None) -> np.ndarray:
        """Define a condição inicial para variável u"""
        if x is None:
            x = self.dom.N
            return np.zeros(x)
        
        elif isinstance(x,np.ndarray):
            return np.zeros(len(x))

    def v_zero(self,
               x: np.ndarray = None) -> np.ndarray:
        """Define a condição inicial para variável v"""
        if x is None:
            x = self.dom.N
            return np.zeros(x)
        
        elif isinstance(x,np.ndarray):
            return np.zeros(len(x))
        
    def calculo_cfl(self):

        cfl = self.dom.dt / self.dom.dx
        if cfl >1 or cfl<0:
            print(f"Instável, cfl = {cfl}.")
        
        return cfl

    def rhs(self,
                h: np.ndarray = None,
                u: np.ndarray = None,
                v: np.ndarray = None
                ) -> np.ndarray:
        """
        Armazenarei nos pontos inteiros a velicidade v e a altura eta.
        E a velocidade u será armazenada no pontos racionais
        """

        def diff_u(vet):
            return (np.roll(vet, 1) - vet)/self.dom.dx

        def diff_h(vet):
            return (vet - np.roll(vet, -1))/self.dom.dx
        
        def mean_v(vet):
            return (np.roll(vet, -1)+ vet)/2.0

        def mean_u(vet):
            return (np.roll(vet, 1)+vet)/2.0

        if h is None:
            h = self.h_zero()

        if u is None:
            u = self.u_zero()

        if v is None:
            v = self.v_zero()

        h_i = self.H*diff_u(u) #discretização da primeira equação

        u_meio =  self.f*mean_v(v)+self.g*diff_h(h)# discretização da segunda equação

        v_i = -self.f*mean_u(u) #Discretização da terceira equação

        return {
            'h' : h_i,
            'u' : u_meio,
            'v' : v_i
        }
  
    def malha_c(self,
                h: np.ndarray = None,
                u: np.ndarray = None,
                v: np.ndarray = None,
                ):


        if h is None:
            h = self.h_zero()

        if u is None:
            u = self.u_zero()

        if v is None:
            v = self.v_zero()


        #first stage of ssprk33 
        step1 =  self.rhs(h,u,v)
        h_1 = h + self.dom.dt*step1['h']
        u_1 = u + self.dom.dt*step1['u']
        v_1 = v + self.dom.dt*step1['v']


        #second stage of ssprk33
        step2 = self.rhs(h_1,u_1,v_1)
        h_2 = 0.75*h +0.25*h_1+0.25*self.dom.dt*step2 ['h']
        u_2 = 0.75*u +0.25*u_1+0.25*self.dom.dt*step2 ['u']
        v_2 = 0.75*v +0.25*v_1+0.25*self.dom.dt*step2 ['v']



        #tird stage of ssprk33
        step3 = self.rhs(h_2,u_2,v_2)
        h_3 = (1/3)*h +(2/3)*h_2 + (2/3)*self.dom.dt*step3['h']
        u_3 = (1/3)*u +(2/3)*u_2 + (2/3)*self.dom.dt*step3['u']
        v_3 = (1/3)*v +(2/3)*v_2 + (2/3)*self.dom.dt*step3['v']


        return {
            'eta_final': h_3 - self.H,
            'h_final': h_3,
            'u_final' : u_3,
            'v_final' : v_3
        }

    def solucao_numerica(self,
                         solucao_h: np.ndarray = None, # condição incial para h
                         solucao_u: np.ndarray = None, #condição inicial para u
                         solucao_v: np.ndarray = None, #condição inicial para v
                         it: int = None, # loops de execução do método numérico
                         modo: str = "malha_c" # modelo de execução
                         ):
        """Calcula a solução de águas rasas após vários instantes."""


            
        if solucao_h is None:
            solucao_h = self.h_zero()
    
        if solucao_u is None:
            solucao_u = self.u_zero()

        if solucao_v is None:
            solucao_v = self.v_zero()

        
        h_final = solucao_h
        u_final = solucao_u
        v_final = solucao_v

            
        for _ in range(it):
            propagacao = self.malha_c(h_final,u_final, v_final)
            h_final = propagacao['h_final']
            u_final = propagacao['u_final']
            v_final = propagacao['v_final'] 
                  
        return {
                'h' : h_final,
                'u' : u_final,
                'v' : v_final,
            }  


if __name__ == "__main__":
    import textwrap
    import dominio

    ###opção
    op = 1
    iteracoes = 2**10

    #### Variáveis
    #N=512; M = 802 #cfl = 0.5
    N=1024; M=900  #cfl = 0.8
    amos = 2
    ruido = False
    first_sample = .2 # paper uses first_sample = 0.2
    Delta_x =  0.09 # paper uses Delta_x = 0.09 end Delta_x = 0.375 for counter-example
    #discretizacao = "godunov_euler"
    #discretizacao = "muscl_ssprk33"
    discretizacao = "malha_c"
    #modo = "malha_c"
    modo = "analitico"
    save = False
    
    ###objetos


    dom = Dominio(N = N, M = M) #cfl = 0.8
   
    sol = SolucaoAguasRasasF(dom)
    cfl = (dom.dt/dom.dx)*np.sqrt(sol.g*sol.H)
    print(f'cfl = {cfl}')


    if op == 1: #teste da solução analítica para η
        for i in range(450):
            y = sol.solucao_numerica(it = i)['h']
            plt.clf()
            plt.plot(dom.x,y)
            plt.title(f'Execução {i+1} de {M} do modelo {discretizacao}.',  fontsize=16)
            plt.legend()
            plt.pause(0.05)
        plt.show()