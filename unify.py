""" This document will unify all the code developed thus far. """
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

import initial_conditions as ci            # initial conditions hub
from domain import Domain as domain       # all domain parameters 



class ForwardSolution:
    ############## constructor #########################
    def __init__(self,
                dom: domain, # package including time and space discretization 
                condition: str = "paper_condition",
                equation: str = "advection"
                ):
    
        self.dom = dom
        self.condition = condition
        self.equation = equation


    ################ initial condition for the water displacement ##########################
    def eta_zero(self, 
                x: np.ndarray = None) -> np.ndarray:
        """This method returns the initial condition for 
        the variable η used to solve the forward equations."""
        init_cond = ci.Function2d()
        if x is None:
            x = self.dom.x
        if self.condition == "square_condition":
            return init_cond.square_condition(x)
        
        else:
            return init_cond.paper_condition(x)


    ################ initial condition for the water velocity ##########################
    def u_zero(self,
            x: np.ndarray = None) -> np.ndarray:
        """This method returns the initial condition for 
        the variable u used to solve the forward equations."""
        if x is None:
            x = self.dom.N
            return np.zeros(x)
        
        elif isinstance(x,np.ndarray):
            return np.zeros(len(x))


    ################ Analytic solution ############################################
   
    def analytic_solution(self,
                          iter: int = None
                          ):
        """Analytic solution to the advection equation assuming velocity equal to 1 
        and linear shallow water equation"""
        if iter is None:
            iter = self.dom.M
        if self.equation == "advection":
        ################ Analytic solution for the advection equation ##################
            return self.eta_zero(self.dom.x - 1 *(iter+1)*self.dom.dt)
        
        ################ Analytic solution for the water displacement ################### 
        elif self.equation == "linear_SWE":
            return 0.5*(self.eta_zero(self.dom.x - (iter+1)*self.dom.dt) \
                        + self.eta_zero(self.dom.x + (iter+1)*self.dom.dt))

        else:
            print("This equation does not have an analytic solution.")

     
    ################# Numeric solution ##################################
    def numeric_solution(self,
                         initial_eta:  np.ndarray = None, # Initial condition for variable eta
                         initial_u: np.ndarray = None, # Initial condition for variable u
                         iter: int = None # Number of iterations 
                         ):

        if iter is None:
            iter = self.dom.M # Standard value for inter

        if initial_eta is None:
            initial_eta = self.eta_zero() # Standard value for eta
    
        if initial_u is None:
            initial_u = self.u_zero() # Standard value for u


        if self.equation == "advection":
            def Lax_Friedrichs(
                               eta: tuple
                                 ):
                """Returns one step of the numerical solution of the advection
                equation using the Lax-Friedrichs method."""
                cfl = self.dom.cfl
                return 0.5*((1 + cfl)*np.roll(eta,1) + (1 - cfl)*np.roll(eta,-1))

            solution = initial_eta

            for _ in range(iter):
                solution = Lax_Friedrichs(solution)

            return solution

        elif self.equation == "linear_SWE":
            def c_grid(
                cond_eta: np.ndarray = None,# located at the center of cell x_i
                cond_u: np.ndarray = None # located at the edge of cell x_{i+0.5}
                ):
    
                def diff_u(vet):
                    return (np.roll(vet,1) - vet)/self.dom.dx

                def diff_eta(vet):
                    return (vet - np.roll(vet,-1))/self.dom.dx

                #first stage of strong stability preserving Runge–Kutta 33  
                eta_1 = cond_eta + self.dom.dt*diff_u(cond_u)
                u_1 = cond_u + self.dom.dt*diff_eta(cond_eta)

                #second stage of strong stability preserving Runge–Kutta 33
                eta_2 = 0.75*cond_eta +0.25*eta_1+0.25*self.dom.dt*diff_u(u_1)
                u_2 = 0.75*cond_u +0.25*u_1+0.25*self.dom.dt*diff_eta(eta_1)

                #third stage of strong stability preserving Runge–Kutta33
                eta_3 = (1/3)*cond_eta +(2/3)*eta_2 + (2/3)*self.dom.dt*diff_u(u_2)
                u_3 = (1/3)*cond_u +(2/3)*u_2 + (2/3)*self.dom.dt*diff_eta(eta_2)


                return {
                        'final_eta' : eta_3,
                        'final_u': u_3
                        }

            propagation = c_grid(initial_eta,initial_u)
                        
            for _ in range(iter+1):
                eta_final = propagation['final_eta']
                u_final = propagation['final_u'] 
                propagation = c_grid(eta_final,u_final)

            return {
                'eta' : eta_final,
                'u'   : u_final 
            }


        else:
            print("Equation not found!")



if __name__ == "__main__":
    from domain import Domain
    import matplotlib.pyplot as plt
    import numpy as np
    import hashlib
    import json
    from pathlib import Path

    ### options

    op = 4
    iterations = 2**3

    #### Variables of the problem

    # N=1025; M = 513 #cfl = 0.5
    N=1024; M = 320 #cfl = 0.8 # Recommended for swe
    #N=512;  M = 160 #cfl = 0.8
    # N=1024; M=256   #cfl = 1 # Recommended for advection

    dom = Domain(N = N, M = M)
    sol = ForwardSolution(dom = dom, 
                    condition = "paper_condition",
                    equation= "linear_SWE" # or  "advection"
                    )

    if op == 20:

        pass

    elif op == 4: # tests of the analytical solution for linear SWE
        sol = ForwardSolution(dom = dom, 
                        condition = "paper_condition",
                        equation= "linear_SWE" 
                        )
        y = sol.numeric_solution()['eta']
        z = sol.analytic_solution()
        plt.plot(dom.x, y, label = "Numeric solution")
        plt.plot(dom.x, z, label = "Analytic solution")
        plt.title("Solution of the linear SWE")
        plt.legend()
        plt.show()

    elif op == 3: # tests of the numerical solution for advection
        sol = ForwardSolution(dom = dom, 
                        condition = "paper_condition",
                        equation= "advection"
                        )
        y = sol.numeric_solution()
        z = sol.analytic_solution()
        plt.plot(dom.x, y, label = "Numeric solution")
        plt.plot(dom.x, z, label = "Analytic solution")
        plt.title("Solution of the advection equation")
        plt.legend()
        plt.show()

    elif op == 2: # tests of the analytical solution for linear SWE
        sol1 = ForwardSolution(dom = dom, 
                        condition = "paper_condition",
                        equation= "linear_SWE" 
                        )
        y = sol1.analytic_solution()
        
        plt.plot(dom.x, y)
        plt.show()

    elif op == 1: # tests of the analytical solution for advection
        sol1 = ForwardSolution(dom = dom, 
                        condition = "paper_condition",
                        equation= "advection" 
                        )
        y = sol1.analytic_solution()
        plt.plot(dom.x, y)
        plt.show()

    