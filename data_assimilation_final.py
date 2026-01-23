"""
Código para assimilação de dados para a equação da advecção.
"""

import numpy as np
import matplotlib.pyplot as plt


class InitialCondition:
    """
    Classe para criar a condição inicial desejada
    """

    def __init__(self, vector):

        self.vector = vector
    
    def paper_initial_condition(self):

        return (1/20)**np.exp((-10*self.vec)**2)

    def u_zero(self):

        if isinstance(self.vector, (np.ndarray, list, tuple)):
            self.vector = np.asarray(self.vector)
            return 0.05*((x >= -1).astype(float) + (x <= 1).astype(float) -1)
        
        else:
            return 0.05*(float(x >= -1) + float(x <= 1) - 1)




class Assimilation:

    def __init__(self, vector):

        self.vector = vector

    

class GraphicConstructor:

    def __init__(self, vector):

        self.y = vector
        
    
    

    