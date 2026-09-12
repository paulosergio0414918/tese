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
                dom: domain = None, # package including time and space discretization 
                condition: str = "paper_condition",
                equation: str = "advection"
                ):
        self.dom = dom
        self.condition = condition
        self.equation = equation
        if (self.dom is None) and (self.equation == "advection"):
            self.dom = Domain(N = 1024, M = 256)

        elif (self.dom is None) and ((self.equation == "linear_SWE") or (self.equation == "nonlinear_SWE")):
            self.dom = Domain(N = 1024, M = 320)

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
                         iter: int = None, # Number of iterations
                         time: float = 2 #execution time of the nonlinear equation 
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

            return {
                    'eta' : solution,
                    }

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

        elif self.equation == "nonlinear_SWE":
            def c_grid(
                    eta: np.ndarray = None, # initial condition to variable eta
                    u: np.ndarray = None, # initial condition to variable u
                    H: float = 1 # average depth
                    ) -> np.ndarray:
                #In the nonlinear, case we need to calculate the time steps and storage. 
                eta_bar = np.array(eta) # Convert the initial condition to a Numpy array
                u_bar = np.array(u) # Convert the initial condition to a Numpy array
                #In the nonlinear, case we need to calculate the time steps and storage. 
                dt = self.dom.cfl*self.dom.dx/np.max(np.abs(u_bar) + np.sqrt(H + eta_bar))
    
                def right_side(eta_bar, u_bar): #obtain the right side of equation           
    
                    def u_center(vector): # put the vector u in centre cell
                        return (np.roll(vector,1) + vector)/2
    
                    def eta_interface(vector): # put the eta vector in interface cell
                        return (np.roll(vector,-1) + vector)/2
    
                    def diff_eta(vector): # calculate the finite difference 
                        return  (np.roll(vector,1) - vector)/self.dom.dx
    
                    def diff_u(vector): # calculate the finite difference 
                        return  (vector - np.roll(vector,-1) )/self.dom.dx
    
    
                    eta_n = diff_eta((H + eta_interface(eta_bar))*u_bar)
                    u_n = diff_u((0.5*u_center(u_bar)*u_center(u)+eta_bar))
    
                    return{
                    'eta_final': eta_n,
                    'u_final': u_n
                        }
    
                
                #first stage of strong stability preserving Runge–Kutta 33 
                eta_1 = eta_bar + dt * right_side(eta_bar, u)['eta_final']
                u_1 = u_bar + dt * right_side(eta_bar, u)['u_final']
                
                #second stage of strong stability preserving Runge–Kutta 33 
                eta_2 = 0.75*eta_bar + 0.25*eta_1 + 0.25*dt*right_side(eta_1, u_1)['eta_final']
                u_2 = 0.75*u_bar + 0.25*u_1 + 0.25*dt*right_side(eta_1, u_1)['u_final']      
                
                #third stage of strong stability preserving Runge–Kutta 33 
                eta_3 = (1/3)*eta_bar + (2/3)*eta_2 + (2/3)*dt*right_side(eta_2, u_2)['eta_final']
                u_3 = (1/3)*u_bar + (2/3)*u_2 + (2/3)*dt*right_side(eta_2, u_2)['u_final']       
                            
    
                return{
                    'eta_final': eta_3,
                    'u_final': u_3,
                    'dt': dt
                }

            propagation = c_grid(eta = initial_eta, u = initial_u)
            dt_vector = [ ]
            flag = 0
            local_time = 0
            while True:
                flag += 1
                local_time += propagation['dt']
                dt_vector.append(propagation['dt'])
                eta_final = propagation['eta_final']
                u_final = propagation['u_final']
                propagation = c_grid(eta = eta_final, u = u_final)
                
                if (flag > 1000) or (local_time > time):
                    #print(f'flag = {flag} and local_time = {local_time}')
                    break   


            return {
                'u'   : u_final,
                'eta' : eta_final,
                'time': local_time,
                'dt_vector' :dt_vector
            }

        else:
            print("Equation not found!")

class BackwardSolution:

    def __init__(self,
                 dom: domain, 
                 n_samples: int = 2,
                 standard_deviation: float = 0.0005,
                 condition: str = "paper_condition",
                 noise: bool = False,
                 equation: str = "advection",
                 Delta_x: float = 0.09,
                 first_sample: float = 0.2
                 ):
        
        self.Delta_x = Delta_x
        self.first_sample = first_sample
        self.equation = equation
        self.n_samples = n_samples
        self.dom = dom
        self.standard_deviation = standard_deviation
        self.condition = condition
        self.noise = noise
        self.sol = ForwardSolution(self.dom, condition=self.condition, equation = self.equation)
        #self.modo = modo
        #self.sample_matrix = None
        #self.sample_matrix_noise = None
        self.cost_vector = []
        self.noise_matrix = np.array([[random.gauss(0, self.standard_deviation) for _ in range(self.dom.M)] for _ in range(self.n_samples)]) # matriz de ordem n_samplesxM
        self.E = np.abs(np.mean(np.sum(self.noise_matrix, axis=1)))
        self.matrix_sample_constructor()
        #self.matrix_sample_noise_constructor()
        self._print_steps_done = True

    def steps_constructor(self,
                            print_info: bool = False):
            
            #FIXME:
            #! For some reason this method fails when delta_x = 0.1 
            
            observation_window = self.dom.x[(self.dom.x>0) & (self.dom.x<2)]
            
            # print_info = True
    
            if print_info:        
                print('---------------------------')
                print(f'Received Delta x  {self.Delta_x}')
                print(f'Received x0 {self.first_sample}')
                print('---------------------------')
                print('')
    
            if (self.first_sample < 0) or (self.first_sample > 2) or (self.Delta_x < 0) or (self.Delta_x > 2): #Remove invalid cases.
                print("Delta_x or the first sample does not belong to the observation window [0, 2]. These values will be replaced.")
                self.first_sample = observation_window[1] # Return the first nonzero term of the  observation window.
                self.Delta_x = self.dom.dx # Returns the best Delta_x for assimilation.
    
            else:
                if np.any(np.isclose(observation_window, self.first_sample)): #x_0 is incompatible with the discretization. 
                    x_ultimo = self.first_sample + (self.n_samples-1)*self.Delta_x
                    if np.any(np.isclose(observation_window,x_ultimo)):#x_j is compatible with the discretization
                        pass 
    
                    else: #x_0 is compatible with the discretization, but xj is not.
                        if self.Delta_x >= self.dom.dx:# if Delta_x > dx, simply set Delta_x to dx
                            Delta_x_local = np.floor(self.Delta_x/self.dom.dx)*self.dom.dx if (np.floor(self.Delta_x/self.dom.dx) != 0) else self.dom.dx
                            x_ultimo = self.first_sample + (self.n_samples-1)*Delta_x_local
                            if np.any(np.isclose(observation_window,x_ultimo)):# if the adaptation falls within to the observation window, it is acceptable.
                                self.Delta_x = Delta_x_local
                            else:# If the adaptation does not falls within the observation window.
                                Delta_x_max = (2 - self.first_sample)/self.n_samples # Biggest delta_x for the first sample provided.
                                if Delta_x_max <= self.dom.dx: # Test the position of the first sample.
                                    self.Delta_x = self.dom.dx
                                    self.first_sample = 2 - (self.n_samples+3)*self.dom.dx
                                else:
                                    Delta_x_local = np.floor(Delta_x_max/self.dom.dx)*self.dom.dx if (np.floor(Delta_x_max/self.dom.dx) != 0) else self.dom.dx
                                    self.Delta_x = Delta_x_local
    
                        else:
                            self.Delta_x = self.dom.dx
                            self.first_sample = 2 - (self.n_samples+3)*self.dom.dx
    
                else:
                    self.first_sample =observation_window[np.argmin(np.abs(observation_window - self.first_sample))]
                    x_ultimo = self.first_sample + (self.n_samples-1)*self.Delta_x
                    #print(f'lest sample = {x_ultimo}')
                    if np.any(np.isclose(observation_window,x_ultimo)):#x_j is compatible with the discretization.
                        pass # The data provided is compatible with the discretization. 
    
                    else: #x_0 is compatible whir the discretization, but xj is not
                        if self.Delta_x >= self.dom.dx:# if Delta_x > dx, simply set Delta_x to dx 
                            Delta_x_local = np.floor(self.Delta_x/self.dom.dx)*self.dom.dx if (np.floor(self.Delta_x/self.dom.dx) != 0) else self.dom.dx
                            x_ultimo = self.first_sample + (self.n_samples-1)*Delta_x_local
                            if np.any(np.isclose(observation_window,x_ultimo)):# if the adaptation falls within to the observation window, it is acceptable.
                                self.Delta_x = Delta_x_local
                            else:# If the adaptation does not falls within the observation window.
                                Delta_x_max = (2 - self.first_sample)/self.n_samples # Biggest delta_x for the first sample provided.
                                if Delta_x_max <= self.dom.dx: # Test the position of the first sample.
                                    self.Delta_x = self.dom.dx
                                    self.first_sample = 2 - (self.n_samples+3)*self.dom.dx
                                else:
                                    Delta_x_local = np.floor(Delta_x_max/self.dom.dx)*self.dom.dx if (np.floor(Delta_x_max/self.dom.dx) != 0) else self.dom.dx
                                    self.Delta_x = Delta_x_local
    
                        else:
                            self.Delta_x = self.dom.dx
                            self.first_sample = 2 - (self.n_samples+3)*self.dom.dx
    
    
    
    
            xj = np.array([self.first_sample + i*self.Delta_x for i in range(self.n_samples)])
            #print(f'vector xj = {xj}')
            position = [np.where(np.isclose(self.dom.x, xj[i]))[0][0] for i in range(self.n_samples)]
            steps = [int(p) for p in position]
            
    
            if print_info:   
                print('---------------------------')
                print(f"Adopted Delta x: {self.Delta_x}")
                print(f"Adopted x0  {self.first_sample}")
                print('---------------------------')
                print('')
                if self.Delta_x > 0.1:
                    console.print("[bold red] The paper requires Delta_x < 0.1 to the assimilation methods to work. [/bold red]")
                self._print_steps_done = False
    
            
            return {
                'steps' : steps,
                'xj': xj 
            }

    #! I'll do some changes in matrix_sample_constructor

    def matrix_sample_constructor(self):
        matrix = np.zeros((self.dom.N, self.dom.M))

        steps = self.steps_constructor()['steps']


        for j in range(self.dom.M):
            solution = sol.numeric_solution(iter = j)['eta']   
            for i in range(self.n_samples):
                matrix[steps[i], j] = solution[steps[i]]

        if self.noise:
            sample_with_noise = np.zeros((self.dom.N, self.dom.M))           
            for i in range(self.n_samples):
                sample_with_noise[steps[i]] = matrix[steps[i],:] + self.noise_matrix[i,:]
            self.sample_matrix = sample_with_noise
            return  sample_with_noise 
        else:
            self.sample_matrix  = matrix
            return matrix 



    def old_matrix_sample_constructor(self):

        """Creates a matrix containing all sample data."""
        
        matrix = np.zeros((self.n_samples, self.dom.M)) 
        steps = self.steps_constructor()['steps']


        for i in range(self.n_samples):
            for j in range(self.dom.M):
                solution = sol.numeric_solution(iter = j)['eta']   
                matrix[i, j] = solution[steps[i]]
        if self.noise:
            sample_with_noise = matrix + self.noise_matrix 
            self.sample_matrix = sample_with_noise
            return  sample_with_noise 
        else:
            self.sample_matrix  = matrix
            return matrix 

    def source_term(self,
                u: np.ndarray = None,
                eta: np.ndarray = None,
                ): # Source term of the finite volume methods applied to the backward system


        x_j = self.steps_constructor()["steps"]
        y_j = self.sample_matrix
        rhs = np.zeros((self.dom.N, self.dom.M))  #right-hand side
        for j  in range(self.dom.M):
            eta_forecast = self.sol.numeric_solution(initial_eta = eta, initial_u = u, iter = j)["eta"]
            for i in range(self.n_samples):
                rhs[x_j[i],j] = eta_forecast[x_j[i]] - y_j[x_j[i],j]
    
        
        return rhs    

    def grad(self,
                cond_eta: np.ndarray = None,
                cond_u: np.ndarray = None
                ):
        #* I think it's okay.
        if self.equation == "advection":
            eta_zero_star = np.zeros(self.dom.N)
            source = self.source_term(cond_eta)
            kappa = self.dom.dt/self.dom.dx
            for i in reversed(range(self.dom.M-1)):
                k = i+1
                grad = 0.5*(np.roll(eta_zero_star,-1)*(1+kappa) - (np.roll(eta_zero_star,1)*(1-kappa))) + kappa*source[:,k]

                eta_zero_star = grad

            return {
                    'eta_grad' : eta_zero_star
                } 


        elif self.equation == "linear_SWE":
            source = self.source_term(u = cond_u, eta = cond_eta )
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
                eta_1 = eta_zero_star - self.dom.dt*(diff_u(u_zero_star)-(1/self.dom.dx) * source[:,k])
                u_1 = u_zero_star- self.dom.dt*diff_eta(eta_zero_star)

                #second stage of ssprk33
                eta_2 = 0.75*eta_zero_star + 0.25*eta_1 - 0.25*self.dom.dt*(diff_u(u_1)-(1/self.dom.dx) * source[:,k-1])
                u_2 = 0.75*u_zero_star + 0.25*u_1 - 0.25*self.dom.dt*diff_eta(eta_1)

                #second stage of ssprk33
                eta_3 = (1/3)*eta_zero_star +(2/3)*eta_2 - (2/3)*self.dom.dt*(diff_u(u_2)-(1/self.dom.dx) * source[:,k-1])
                u_3 = (1/3)*u_zero_star +(2/3)*u_2 - (2/3)*self.dom.dt*diff_eta(eta_2)

                eta_zero_star = eta_3
                u_zero_star = u_3

            

            return {
                    'eta_grad' : eta_zero_star ,
                    'u_grad': u_zero_star
                } 

    def reconstruction_error(self,vet):
        return np.linalg.norm(vet - self.sol.eta_zero())/np.linalg.norm(self.sol.eta_zero())
  
    def gradient_descent(self,
                              it:int = 10):
        """Calculo do gradiente descendente considerando n=it iterações"""

        from tqdm import tqdm
        final_solution_eta = np.zeros(self.dom.N) # initial eta
        final_solution_u = np.zeros(self.dom.N) # initial u
        error = []
        cost = []

        if self.equation == "advection":
            for _ in tqdm(range(it)):
                grad_eta_local = self.grad(cond_eta = final_solution_eta)['eta_grad']
                final_solution_eta = final_solution_eta - 0.1*grad_eta_local
                error.append(self.reconstruction_error(final_solution_eta))
                cost.append(self.assimilation_cost(final_solution_eta))

            return {
                    'eta_final' : final_solution_eta, # eta após it execuções do gradiente descendente con learning rate fixo
                    'error' : error, # Erro de reconstrução de cada passo do gradiente descendente
                    'cost': cost, # funcional cost de cada passo do gradiente descendente
                }
        elif self.equation == "linear_SWE":

            for _ in tqdm(range(it)):
                grad = self.grad(cond_eta = final_solution_eta, cond_u = final_solution_u)
                final_solution_eta = final_solution_eta - 0.1*grad["eta_grad"]
                final_solution_u = final_solution_u - 0.1*grad["u_grad"]
                error.append(self.reconstruction_error(final_solution_eta))
                cost.append(self.assimilation_cost(final_solution_eta))

            return {
                    'eta_final' : final_solution_eta, # eta após it execuções do gradiente descendente con learning rate fixo
                    'u_final': final_solution_u, # u após it execuções do gradiente descendente con learning rate fixo
                    'error' : error, # Erro de reconstrução de cada passo do gradiente descendente
                    'cost': cost # funcional custo de cada passo do gradiente descendente
                }
    #FIXME: The optimized  gradient descent  does notwork.
    #! For some reason this method does not converge. 
    #! eu estou atualizando o eta?
    def optimized_gradient_descent(self,
                                it:int = 10):
        """Calculo do gradiente descendente considerando n=it iterações"""

        def reconstruction_error(vet):
            return np.linalg.norm(vet - self.sol.eta_zero())/np.linalg.norm(self.sol.eta_zero())

        def grad_eta(eta): 
            return self.grad(cond_eta = eta, cond_u = final_solution_u)['eta_grad']

        from tqdm import tqdm
        from scipy.optimize import line_search
        final_solution_eta = np.zeros(self.dom.N) #initial eta
        final_solution_u = np.zeros(self.dom.N) # initial u
        error = []
        cost = []
        alpha = []
           
    
        if self.equation == "advection":
            for i in tqdm(range(it)):
                grad_eta_local = self.grad(cond_eta = final_solution_eta)['eta_grad']
                optim = line_search(self.assimilation_cost, grad_eta, final_solution_eta, -grad_eta_local ) 
                if optim[0] is None:
                    alpha_i = 0.1
                    print(f"Não houve otimização do passo na iteração {i}")
                else:
                    alpha_i = optim[0]

                final_solution_eta = final_solution_eta - alpha_i*grad_eta_local
                error.append(reconstruction_error(final_solution_eta))
                cost.append(self.assimilation_cost(final_solution_eta))
                alpha.append(alpha)

            return {
                    'eta_final' : final_solution_eta, # eta após it execuções do gradiente descendente con learning rate fixo
                    'error' : error, # Erro de reconstrução de cada passo do gradiente descendente
                    'cost': cost, # funcional cost de cada passo do gradiente descendente
                    'alpha': alpha, # passo do gradiente descendente para ser aproveitado posteriormente
                }


        elif self.equation == "linear_SWE":
            for i in tqdm(range(it)):

                grad_eta_local = grad_eta(eta = final_solution_eta) 
                grad_u = self.grad(cond_eta = final_solution_eta, cond_u = final_solution_u)['u_grad']         
                optim = line_search(self.assimilation_cost, grad_eta, final_solution_eta, -grad_eta_local ) 
                if optim[0] is None:
                    alpha_i = 0.1
                    print(f"Não houve otimização do passo na iteração {i}")
                else:
                    alpha_i = optim[0]
                final_solution_eta = final_solution_eta - alpha_i*grad_eta_local
                final_solution_u = final_solution_u - 0.1*grad_u
                error.append(reconstruction_error(final_solution_eta))
                cost.append(self.assimilation_cost(final_solution_eta))
                alpha.append(alpha)

            return {
                    'eta_final' : final_solution_eta, # eta após it execuções do gradiente descendente con learning rate fixo
                    'u_final': final_solution_u, # u após it execuções do gradiente descendente con learning rate fixo
                    'error' : error, # Erro de reconstrução de cada passo do gradiente descendente
                    'cost': cost, # funcional cost de cada passo do gradiente descendente
                    'alpha': alpha, # passo do gradiente descendente para ser aproveitado posteriormente
                }




    def assimilation_cost(self,
                              eta: np.ndarray = None,
                              ):
        """Retorna o custo de assimilação para cada iteração."""
        steps = self.steps_constructor()['steps'] #gera o indice onde estão as amostras no vetor de assiilação
        diff= np.zeros((self.n_samples, self.dom.M))# vai receber as diferenças internas do custo
        y_j = self.sample_matrix
        eta_forecast = np.zeros((self.dom.N, self.dom.M))
        for i in range(self.dom.M): # loop para construir a diferença presente no custo
            eta_f = self.sol.numeric_solution(initial_eta=eta, initial_u=np.zeros(self.dom.N), iter=i)['eta'] # constroi o eta^f dada a condicao tomando u = 0
            eta_forecast[:,i] = eta_f
            # Atualiza solução
        for j in range(self.n_samples):
            diff[j,:] = (eta_forecast[steps[j],:] - y_j[steps[j],:])**2
        sum_diff = np.sum(diff, axis=0) # Returns a vector of size self.M containing the sum over all n_samples columns.

        def trapezoidal_rule(x): # integral using trapezoidal rule
            s=0
            n = len(x)
            for i in range(1,n-1,1):
                s += x[i]
            return (x[0] + 2*s + x[-1])*self.dom.dt/2

        return 0.5 * trapezoidal_rule(sum_diff ) #Returns the numerical integral 







if __name__ == "__main__":
    from domain import Domain
    import textwrap
    import matplotlib.pyplot as plt
    import numpy as np
    import hashlib
    import json
    from pathlib import Path

    ### options

    op = 9
    iterations = 2**3

    #### Variables of the problem

    #N=1025; M = 513 #cfl = 0.5
    #N=1024; M = 320 #cfl = 0.8 # Recommended for swe
    #N=512;  M = 160 #cfl = 0.8
    N=1024; M=256   #cfl = 1 # Recommended for advection
    
    amos = 2
    noise = False
    first_sample = 0.2 # paper uses first_sample = 0.2
    Delta_x =  0.09 # paper uses Delta_x = 0.09 end Delta_x = 0.375 for counter-example
    equation = "linear_SWE" # or  "advection" or "nonlinear_SWE" or "linear_SWE" 

    dom = Domain(N = N, M = M)
    sol = ForwardSolution(dom = dom, 
                    condition = "paper_condition", # or "square_condition"
                    equation= "advection" # or  "advection" or "nonlinear_SWE" or "linear_SWE" 
                    )
    assimilation = BackwardSolution(dom = dom,
                        n_samples = amos,
                       noise = noise,
                       first_sample = first_sample, 
                       Delta_x =  Delta_x,
                       equation = equation
                         ) 

    if op == 20: # Reproducing the paper's figure 3 
        sol1 = ForwardSolution(dom = dom, 
                        condition = "paper_condition",
                        equation= "linear_SWE" 
                        )
        sol2 = ForwardSolution(dom = dom, 
                        condition = "paper_condition",
                        equation= "nonlinear_SWE" 
                        )
        
        y = sol1.numeric_solution()['eta']
        z = sol2.numeric_solution()['eta']
        plt.plot(dom.x, y, label = "Linear SWE")
        plt.plot(dom.x, z, label = "Nonlinear SWE")
        plt.title("""Right-going part of the linear and nonlinear SWE solutions at the observation time T = 2. Note
the significant difference between the two solutions""")
        plt.legend()
        plt.show()

    elif op == 9: #assimilation graphic
        ass_local = BackwardSolution(dom = dom, n_samples = amos,
                               noise = noise,
                               first_sample = first_sample,
                               equation =  equation,
                               Delta_x =  Delta_x  )
        result = ass_local.gradient_descent(it=iterations)
        caso = ass_local.steps_constructor()
        steps = caso['xj']

        plt.ylim(-0.025, 0.06) # y limit
        plt.xlim(-1.5, 1.5) # x limit
        plt.plot(dom.x, result['eta_final'], color = "black", linestyle='-', label = 'phi^(f)(x) assimilada' )
        plt.plot(dom.x, sol.eta_zero(dom.x), color = "black", linestyle='--', label = 'phi^(t)(x) realidade')
        for px in steps:# destacar os pontos de amostragem
            plt.plot([px, px], [-0.001, 0.001], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        if noise:
            plt.title(f'Execução de {iterations} iterações utilizando {amos} amostras com Delta x =  {ass_local.Delta_x}. com ruido'  )
        else:
            plt.title(f'Execução de {iterations} iterações utilizando {amos} amostras com Delta x =  {ass_local.Delta_x}. sem ruido')
        plt.legend()
        texto = (
            "Este gráfico mostra a comparação entre a solução assimilada (linha contínua) "
            f"e a realidade (linha tracejada) para a equação {equation} considerando {iterations}"
            f"iterações do gradiente descendente. Neste experimento consideramos {amos} pontos"
            f"amostrais sendo o primeiro cituado em x_o = {first_sample} e igualmente espaçados " 
            f"com Δx = {Delta_x} O numero de amostras para este experimento é  Os traços verticais "
            "vermelhos indicam os pontos de amostragem utilizados pelo método de assimilação. "
        )

        # Quebra o texto em linhas de até ~90 caracteres
        texto_formatado = "\n".join(textwrap.wrap(texto, width=90))

        # Abre espaço embaixo para o texto caber (0.30 = 30% da figura reservada)
        plt.subplots_adjust(bottom=0.30)

        # Insere o texto na margem inferior, usando coordenadas da FIGURA
        plt.figtext(
            0.5, 0.02,                 # x=centro, y=2% acima da base da figura
            texto_formatado,
            ha='center', va='bottom',
            fontsize=9, style='italic', color='dimgray',
            wrap=True
            )
        #plt.savefig('assimilacao.png')iteracoes
        plt.show()

    elif op == 8: #teste de otimizacao

        erro_otimizado = assimilation.optimized_gradient_descendent(it=iterations)['error']
        erro = assimilation.gradient_descendent(it=iterations)['error']

        plt.ylabel('||ϕ^(t)-ϕ^(n)||/||ϕ^(t)||')
        plt.xlabel('Número de iterações')
        plt.yscale('log')
        plt.scatter([i+1 for i in range(iterations)], erro_otimizado , lw = 0.5, label = 'erro otimizado' )
        plt.scatter([i+1 for i in range(iterations)], erro , lw = 0.5, label = 'erro fixo' )
        plt.title(f'Convergencia do custo após {iterations} iterações considerando Δx =  {assimilation.Delta_x}.')
        plt.legend()
        plt.show()

    elif op == 7: # print one sample

        x = dom.t
        y = assimilation.sample_matrix[0,:]
        plt.plot(x,y)
        plt.show()
         
    elif op == 6: # print steps of sample
        case1 = assimilation.steps_constructor()
        print(f'observation points {case1['steps']}')
        print(f'Vector xj = {case1['xj']}')

    elif op == 5: # Tests of the numerical solution for nonlinear SWE
        sol = ForwardSolution(dom = dom, 
                        condition = "paper_condition",
                        equation= "nonlinear_SWE" 
                        )
        y = sol.numeric_solution()['eta']
        plt.plot(dom.x, y, label = "Numeric solution")
        plt.title("Solution of the nonlinear SWE")
        plt.legend()
        plt.show()

    elif op == 4: # tests of the numerical solution for linear SWE
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

    elif op == 0:
        dom_local = Domain(N = N, M = M)

        assimilation_local = BackwardSolution(dom = dom_local,
                        n_samples = amos,
                        noise = noise,
                        first_sample = first_sample, 
                        equation = "advection",
                        Delta_x =  Delta_x
        )

        samples = assimilation_local.matrix_sample_constructor()
        #source = assimilation_local.source_term(eta = np.ones(N))
        print(assimilation_local.steps_constructor()['steps'])
        print(samples.shape)
        np.savetxt('minha_matriz.csv', samples, delimiter=',', fmt='%.10f')
        '''x = np.linspace(-4,4,N)
        y =  assimilation_local.grad()['eta_grad']
        plt.plot(x,y)
        plt.show()'''