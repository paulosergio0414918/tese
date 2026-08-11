# SWE with c-grid
import numpy as np
from rich.traceback import install
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
install()

import condicoes_iniciais as ci
from dominio import Dominio

class Linear_SWE:
    def __init__(self,
                dom: Dominio,
                condicao: str = "condicao_paper"
                 ):
        self.dom = dom
        self.condicao = condicao
        if condicao is None:
            pass

    
    def delta_x(self,
                u: np.ndarray,
                ):
        "Criate de difference operator to change coodenates"
        delta_x =  (u - np.roll(u, 1))/self.dom.dx

        return delta_x



if __name__ == "__main__":
    
    swe = Linear_SWE()
    op = 0

    if op == 0: #test of delta_x operator
        x = [1,2,3,4]
        print(x)
        print(swe.delta_x(x))

