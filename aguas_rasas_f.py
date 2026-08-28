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

class SolucaoAguasRasas:
    """ Solucao analitica e numerica da equação de águas rasas."""

    def __init__(self,
                 dom: Dominio,
                 condicao: str = "condicao_paper",
                 H: float = 1,
                 f: float = 7.292e-05,
                 g: float = 9.81
                 ):
        
        self.dom = dom
        self.condicao = condicao
        self.variacao_energia = 0
        self.H = H
        self.f = f
        self.g = g
    
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

    def malha_c(self,
                eta: np.ndarray = None,
                u: np.ndarray = None,
                v: np.ndarray = None
                ) -> np.ndarray:
        """
        Armazenarei nos pontos inteiros a velicidade v e a altura eta.
        E a velocidade u será armazenada no pontos racionais
        """
        if eta is None:
            eta = self.eta_zero()

        if u is None:
            u = self.u_zero()

        if v is None:
            v = self.v_zero()

        h_i = self.H*(np.roll(u,1)-u)/self.dom.dx #discretização da primeira equação

        u__meio = (self.f*(np.roll(v,-1)-v)+self.g*(eta - np.roll(eta,-1)))/self.dom.dx # discretização da segunda equação

        v_i = self.f*(np.roll(u,1)-u)/self.dom.dx #Discretização da terceira equação

        #integração temporal SSPRK33
        
