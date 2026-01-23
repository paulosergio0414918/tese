import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import math

############ parâmetros ###################
option = 1
#------------------------------------
#se option = 1 <-- esta opção apresenta o gráfico da assimilação
#se option = 2 <-- esta opção apresenta o gráfico do custo de assimilação
#se option = 3 <-- refazer os gráficos da figura 7 do artigo kevlahan
#se option = 4 <-- só os graficos da opção 3 
#se option = 5 <-- grafico da transfomada de Fourier da condição inicial
#se option = 6 <-- grafico da condição inicial
#se option = 7 <-- refazer a figura 7 do artigo kevlahan p/ todas as condições iniciais
#se option = 8 <-- grafico de todas as condições iniciais

########### condição inicial ##############
condicao = 2 
#-----------------------------------------
#condicao = 2 #<-- condição inicial do paper
#condicao = 5 #<-- condição inicial degrau


n_testes = 2
n_amostras = 3

################ variables###############
#the text use large periodic domain [-L,L], where L = 3 
# and number of points N=1024.
L = 3
N = 4056
# The author use T = 2 for observation time
a = 1 
T = 2
# Sample window needs to be greater than 0 end less than 2
Cost = 0

def fourier(k:float) -> float: #trasformada de forier da condição do paper
  return 1/(200*np.sqrt(2)*np.exp((k**2)/400))


############## Initial condition ###########
if condicao == 0:    
      def u_zero(x:float) -> float:
        return (1/20)*np.exp(-1*x**2)

elif condicao == 1: 
  def u_zero(x:float) -> float:
    return (1/20)*np.exp(-10*x**2)

elif condicao == 2:
  def u_zero(x:float) -> float:#condição inicial do paper
    return (1/20)*np.exp(-100*x**2) 
   
#elif condicao == 3:
  #def u_zero(x:float) -> float: 
    #return (1/20)*np.exp(-(10*x)**2)
elif condicao == 3:
  def u_zero(x:float) -> float:
    return (1/20)*np.exp(-1000*x**2)
  
elif condicao == 4:
  def u_zero(x:float) -> float:
    return (1/20)*np.exp(-5000*x**2) 
  
elif condicao == 5:
  def u_zero(x):
    if isinstance(x, (np.ndarray, list, tuple)):
        x = np.asarray(x)
        return 0.05*((x >= -1).astype(float) + (x <= 1).astype(float) - 1)
    else:
        return 0.05*(float(x >= -1) + float(x <= 1) - 1)

   
def u_g(x:float) -> float: # first guess to initial condition
    return np.zeros(len(x))

##############  Method #########

delta_x = 2*L/N # to evaluate the CFL condition.
M = N/3
delta_t = T/M
CFL = a* delta_t/delta_x

if CFL <= 1:
    print(f'The method could be estable, because CFL = {CFL:.2f}.')

else:
    print(f'The method never be estable, because CFL = {CFL:.2f}.')

kappa1 = 0.5 * (1-CFL)
kappa2 = 0.5 * (1+CFL)

def Lax_Friedrichs(x):
  u_forward = np.roll(x,1)
  u_backward = np.roll(x,-1) 
  v = kappa2*u_forward+kappa1*u_backward
  return v

def Lax_Friedrichs_transpose(x):
  u_forward = np.roll(x,1)
  u_backward = np.roll(x,-1) 
  v = kappa1*u_forward+kappa2*u_backward
  return v

def functional_cost(approximate_solution):
    cost = np.zeros(N)
    if option == 3:
      for j in range(n_amostras_local):
        cost += 0.5*(matriz_de_amostras_local[:,j]-approximate_solution)**2
    else:
      for j in range(n_amostras):
        cost += 0.5*(matriz_de_amostras[:,j]-approximate_solution)**2
    return cost.sum()


def convergencia(real_solution, approximate_solution):
    if len(real_solution) != len(approximate_solution):
        print('The solutions do not have the same length. Correct this!')
        return None 
    else:
        return np.linalg.norm(np.array(real_solution) - np.array(approximate_solution)) / np.linalg.norm(real_solution)

real_solution = [1,2,3,4,5,6]
approximate_solution = [1,2,3,4,5,6]
print(convergencia(real_solution, approximate_solution))  # Saída correta: 0.0

###########collecting samples ################

matriz_de_amostras = np.zeros((N,n_amostras))

for i in range(N):
  matriz_de_amostras[i,:] = u_zero((i-N/2)*delta_x)


############ Ploting ###############################

x_values = np.linspace(-L,L,N)
d = np.zeros((N,n_amostras))
gess = u_g(x_values)
gess_analitico = u_g(x_values)


if option == 1:
  error = []
  OrdemDeConvergencia = []
  for j in range(n_testes):
      #ploting the graph
      plt.clf()
      #plt.ylim(-1, 2) # y limit
      #plt.xlim(-2, 2) # x limit
      plt.ylim(-0.025, 0.06) # y limit
      plt.xlim(-1.5, 1.5) # x limit
      plt.plot(x_values, gess, lw = 1, color = 'red', label = '$\phi^{(f)}(x)$' )
      plt.plot(x_values, u_zero(x_values), lw = 1, color = 'blue',label = '$\phi^{(t)}(x)$' )
      plt.title(f'Execução {j+1} de {n_testes} utilizando {n_amostras} amostras com $\Delta x = $ {delta_x}.')
      plt.legend()
      plt.pause(0.1)
      # computing the cost
      Cost1 = functional_cost(gess)
      if Cost1 > Cost:
          print(f'Iteração{j+1}:O funcional custo almentou para = {Cost1}.')
      elif Cost1 < Cost:
          print(f'Iteração{j+1}:O funcional custo diminuiu para = {Cost1}.')
      else:
          print(f'Iteração{j+1}:O funcional custo não alterou')
      error += [Cost1]
      OrdemDeConvergencia += [convergencia(gess,u_zero(x_values))]
      Cost = Cost1
      # computing the discrepance
      for i in range(n_amostras):
          d[:,i] = gess - matriz_de_amostras[:,i]
      # computing the gradient
      grad = d[:,i]
      for i in range(n_amostras-1, 0, -1):
          grad = Lax_Friedrichs_transpose(grad)+d[:,i-1]
      # update the inition condition
      gess = gess-0.1*grad
  plt.show()
  #plt.clf()
  plt.yscale('log')
  plt.xscale('log')
  plt.scatter([i for i in range(n_testes)], OrdemDeConvergencia, lw = 1, color = 'blue',label = '$\mathcal{J}(\phi^{(f)}(x))$ em cada iteração' )
  plt.legend()
  plt.show()

elif option == 2:
  norm = []
  error = [1]
  for j in range(n_testes):
      if j == 0:
        local_gess = u_g(x_values)
        Cost1 = functional_cost(gess)
      else:
        error += [functional_cost(gess)/Cost1]
      # computing the discrepance
      for i in range(n_amostras):
          d[:,i] = gess - matriz_de_amostras[:,i]
      # computing the gradient
      grad = d[:,i]
      for i in range(n_amostras-1, 0, -1):
          grad = Lax_Friedrichs_transpose(grad)+d[:,i-1]
      # update the inition condition
      #modificarei apenas daqui
      gamma_local = 1/n_testes
      local_error = 100
      for z in range(n_testes):
        local_gess = local_gess-gamma_local*(z+1)*grad
        local_error1 = functional_cost(local_gess)
        if local_error>local_error1:
           local_error = local_error1
           gamma = gamma_local*(z+1)
      gess = gess-gamma*grad
      local_gess = gess
      #até aqui, de ser merda apage este intervalo e desmarque a linha abaixo
      #gess = gess-0.1*grad
      norm += [np.linalg.norm(u_zero(x_values)-gess)/np.linalg.norm(u_zero(x_values))]
  #plt.ylim(-0.025, error[0]) # y limit
  #plt.xlim(-1, n_testes+1) # x limit
  plt.yscale('log')
  plt.xscale('log')
  plt.xlabel('Iteração $n$')
  plt.plot([i for i in range(n_testes)], error, label = '$\mathcal{J}^{(n)}/\mathcal{J}^{(0)}$')
  plt.plot([i for i in range(n_testes)], norm, color = 'black', label = '$||\phi^{(t)} - \phi^{(n)}||_2/||\phi^{(t)}||_2 $')
  plt.title(f'Gráfico do custo funcional após {n_testes} iterações e {n_amostras} amostras')
  plt.legend()
  plt.show()

elif option == 3:
   Deltax = []
   final_norm = []
   n_amostras_local = 2
   n_testes_local = 300
   distance = 1
   d_local = np.zeros((N,n_amostras_local))
   matriz_de_amostras_local = np.zeros((N,n_amostras_local))
   for i in range(N):
      matriz_de_amostras_local[i,:] = u_zero((i-N/2)*delta_x)
   for c in range(180):
    print(f'Iniciando o processo {c}.')
    Deltax += [(c+1)*delta_x]
    norm = [0]
    error = [1]     
    for j in range(n_testes_local):
          if j == 0:
            local_gess = u_g(x_values)
            local_Cost1 = functional_cost( local_gess)
          else:
            error += [functional_cost(local_gess)/local_Cost1]
          # computing the discrepance
          for i in range(n_amostras_local):
              d_local[:,i] = np.copy(local_gess - matriz_de_amostras[:,i])
          # computing the gradient
          grad_local = np.copy(d_local[:,1])
          for i in range(n_amostras_local-1, 0, -1):
            for _ in range(distance):
                grad_local = Lax_Friedrichs_transpose(grad_local)
            grad = grad_local+d_local[:,0]
          # update the inition condition
          local_gess = local_gess-0.1*grad
          #modificarei apenas daqui
          #gamma_local = 1/n_testes_local
          #local_error = 100
          #for z in range(n_testes_local):
            #local_gess = local_gess-gamma_local*(z+1)*grad
            #local_error1 = functional_cost(u_zero(x_values), local_gess)
            #if local_error>local_error1:
              #local_error = local_error1
              #gamma = gamma_local*(z+1)
          #gess = gess-gamma*grad
          #local_gess = gess
          #até aqui
          norm += [np.linalg.norm(u_zero(x_values)-local_gess)/np.linalg.norm(u_zero(x_values))]
    distance += 1
    final_norm += [norm[-1]]
    plt.clf()
    plt.xlabel('$\Delta x$')
    plt.ylabel('$||\phi^{(t)} - \phi^{(n)}||_2/||\phi^{(t)}||_2 $')
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.ylim(0.02,0.04) # y limit
    #plt.xlim(-1, n_testes+1) # x limit
    plt.plot(Deltax, final_norm, lw = 1, color = 'blue',label = '$\phi(x)$' )
    plt.title(f'Execução {(c+1)} de 180.')
    #plt.legend()
    plt.pause(0.1)
   plt.show()
   #print(final_norm) 

elif option == 4:
  Deltax = []
  final_norm = []
  n_amostras_local = 2
  n_testes_local = 1000
  distance = 82
  d_local = np.zeros((N,n_amostras_local))
  matriz_de_amostras_local = np.zeros((N,n_amostras_local))
  for i in range(N):
    matriz_de_amostras_local[i,:] = u_zero((i-N/2)*delta_x)
  error_local = []
  distancia = 1
  for j in range(n_testes_local):
      if j == 0:
        local_gess = u_g(x_values)
      else:
        local_gess = local_gess-0.1*grad
      #ploting the graph
      plt.clf()
      plt.ylim(-0.025, 0.06) # y limit
      plt.xlim(-1, 1) # x limit
      plt.xlabel('x')
      plt.plot(x_values, local_gess, lw = 1, color = 'red', label = '$x(t)$' )
      plt.plot(x_values, u_zero(x_values), lw = 1, color = 'blue',label = '$\phi(x)$' )
      plt.title(f'$\Delta x$ = {distance*delta_x}.')
      plt.legend()
      plt.pause(0.1)
      # computing the cost
      Cost1 = functional_cost(local_gess)
      Cost0 = Cost1
      if Cost1 > Cost:
          print(f'Na iteração {j} o funcional custo almentou para = {Cost1}.')
      elif Cost1 < Cost:
          print(f'Na iteração {j} o funcional custo diminuiu para = {Cost1}.')
      else:
          print(f'Na iteração {j} o funcional custo não alterou')
      error_local += [Cost1/Cost0]
      Cost = Cost1
      #final_error = 
      # computing the discrepance
      for i in range(n_amostras_local):
          d_local[:,i] = local_gess - matriz_de_amostras_local[:,i]
      # computing the gradient
      grad_local = d_local[:,1]
      for _ in range(distance):
        grad_local = Lax_Friedrichs_transpose(grad_local)
      grad = Lax_Friedrichs_transpose(grad_local)+d_local[:,0]
      # update the inition condition
      #gess = gess-0.1*grad
  print('Salvar a imagem.')
  plt.pause(30)
  plt.clf()
  plt.yscale('log')
  plt.xscale('log')
  plt.ylabel('$J^{(n)}/J^{(0)}$ em cada iteração')
  plt.scatter([i for i in range(n_testes_local)], error_local, lw = 0.5, color = 'blue',label = '$J^{(n)}/J^{(0)}$ em cada iteração' )
  plt.legend()
  plt.show()     
  
elif option == 5: #imprimi a trasformada de fourier
  x = [i-40 for i in range(80)]
  y = [fourier(x[i]) for i in range(80)]
  plt.ylim(0, 0.004) # y limit
  plt.xlim(-40, 40) # x limit
  plt.ylabel('E(k)')
  plt.xlabel('Número de onda k')
  plt.plot(x, y, color = 'blue' )
  plt.legend()
  plt.show() 

elif option == 6: #imprimi a condição inicial do paper
  x = np.linspace(-3,3,1024)
  y = [u_zero(x[i]) for i in range(1024)]
  #print(f'comprimento de y = {len(y)}')
  #print(f'comprimento de x = {len(x)}')
  plt.ylim(0, 0.1) # y limit
  plt.xlim(-3, 3) # x limit
  plt.ylabel('$\phi(x)$')
  plt.xlabel('x')
  plt.plot(x, y, color = 'blue',label = 'Condição inicial verdadeira' )
  plt.legend()
  plt.show()

elif option == 7: #refazer a figura 7 do artigo kevlahan p/ todas as condições iniciais
  for jj in range(6):
    if  jj == 0:    
      def u_zero(x:float) -> float:
        return (1/20)*np.exp(-1*x**2)
    elif jj == 1: 
      def u_zero(x:float) -> float:
        return (1/20)*np.exp(-10*x**2)
    elif jj == 2:
      def u_zero(x:float) -> float:
        return (1/20)*np.exp(-100*x**2)
    elif jj == 3:
      def u_zero(x:float) -> float:
        return (1/20)*np.exp(-1000*x**2)
    elif jj == 4:
      def u_zero(x:float) -> float:
        return (1/20)*np.exp(-5000*x**2)
    elif jj == 5:
      def u_zero(x):
        if isinstance(x, (np.ndarray, list, tuple)):
          x = np.asarray(x)
          return 0.05*((x >= -1).astype(float) + (x <= 1).astype(float) - 1)
        else:
          return 0.05*(float(x >= -1) + float(x <= 1) - 1)
    Deltax = []
    final_norm = []
    n_amostras_local = 2
    n_testes_local = 750
    distance = 1
    d_local = np.zeros((N,n_amostras_local))
    matriz_de_amostras_local = np.zeros((N,n_amostras_local))
    for i in range(N):
      matriz_de_amostras_local[i,:] = u_zero((i-N/2)*delta_x)
    for c in range(50):
      print(f'Iniciando o processo {c} de passo {jj}.')
      Deltax += [(c+1)*delta_x]
      norm = [0]
      error = [1]     
      for j in range(n_testes_local):
            if j == 0:
              local_gess = u_g(x_values)
              local_Cost1 = functional_cost(local_gess)
            else:
              error += [functional_cost(local_gess)/local_Cost1]
            # computing the discrepance
            for i in range(n_amostras_local):
                d_local[:,i] = np.copy(local_gess - matriz_de_amostras[:,i])
            # computing the gradient
            grad_local = np.copy(d_local[:,1])
            for i in range(n_amostras_local-1, 0, -1):
              for _ in range(distance):
                  grad_local = Lax_Friedrichs_transpose(grad_local)
              grad = grad_local+d_local[:,0]
            # update the inition condition
            local_gess = local_gess-0.1*grad
            norm += [np.linalg.norm(u_zero(x_values)-local_gess)/np.linalg.norm(u_zero(x_values))]
      distance += 1
      final_norm += [norm[-1]]
    if jj == 0:
      Final_error = np.zeros((len(final_norm), 6))
      Final_error[:,jj] = final_norm
    else:
      Final_error[:,jj] = final_norm 
  plt.clf()
  plt.xlabel('$\Delta x$')
  plt.ylabel('$||\phi^{(t)} - \phi^{(n)}||_2/||\phi^{(t)}||_2 $')
  plt.plot(Deltax, Final_error[:,0], lw = 1, color = 'blue',label = '$\phi(x)$' )
  plt.plot(Deltax, Final_error[:,1], lw = 1, color = 'red',label = '$\phi(x)$' )
  plt.plot(Deltax, Final_error[:,2], lw = 1, color = 'green',label = '$\phi(x)$' )
  plt.plot(Deltax, Final_error[:,3], lw = 1, color = 'black',label = '$\phi(x)$' )
  plt.plot(Deltax, Final_error[:,4], lw = 1, color = 'gray',label = '$\phi(x)$' )
  plt.plot(Deltax, Final_error[:,5], lw = 1, color = 'yellow',label = '$\phi(x)$' )
  plt.title(f'Execução {c+1} de 50.')
  #plt.legend()
  plt.pause(0.1)
  plt.show()
  #print(final_norm) 
  #  

elif option == 8: # grafico de todas as condições iniciais
  def degrau(x):
    if isinstance(x, (np.ndarray, list, tuple)):
        x = np.asarray(x)
        return 0.05*((x >= -1).astype(float) + (x <= 1).astype(float) - 1)
    else:
        return 0.05*(float(x >= -1) + float(x <= 1) - 1)
    
  def u_zero_0(x:float) -> float:
    return (1/20)*np.exp(-1*x**2)
  
  def u_zero_1(x:float) -> float:
    return (1/20)*np.exp(-10*x**2)
  
  def u_zero_2(x:float) -> float:
    return (1/20)*np.exp(-100*x**2)
  
  def u_zero_3(x:float) -> float:
    return (1/20)*np.exp(-1000*x**2)
  
  def u_zero_4(x:float) -> float:
    return (1/20)*np.exp(-5000*x**2)
  
  x1 = np.linspace(-3,3,1024)
  y0 = u_zero_0(x1) 
  y1 = [u_zero_1(x1[i]) for i in range(1024)]
  y2 = [u_zero_2(x1[i]) for i in range(1024)]
  y3 = [u_zero_3(x1[i]) for i in range(1024)]
  y4 = [u_zero_4(x1[i]) for i in range(1024)]
  y5 = [degrau(x1[i]) for i in range(1024)]
  #print(f'comprimento de y = {len(y)}')
  #print(f'comprimento de x = {len(x)}')
  plt.ylim(0, 0.1) # y limit
  plt.xlim(-3, 3) # x limit
  plt.ylabel('$\phi^{(t)}(x)$')
  plt.xlabel('x')
  plt.plot(x1, y0, color = 'blue',label = '$k_{max}=4, \Delta x < 0.8$' )
  plt.plot(x1, y1, color = 'red',label = '$k_{max}=12, \Delta x < 0.3$' )
  plt.plot(x1, y2, color = 'green',label = '$k_{max}=30, \Delta x < 0.1$' )
  plt.plot(x1, y3, color = 'black',label = '$k_{max}=64, \Delta x < 0.05$' )
  plt.plot(x1, y4, color = 'gray',label = '$k_{max}=73, \Delta x < 0.04$' )
  plt.plot(x1, y5, color = 'yellow',label = '$k_{max}=\infty, \Delta x < 0$' )
  plt.legend()
  plt.show()

elif option == 9:
  error = []
  OrdemDeConvergencia = []
  for j in range(n_testes):
      #ploting the graph
      plt.clf()
      #plt.ylim(-1, 2) # y limit
      #plt.xlim(-2, 2) # x limit
      plt.ylim(-0.025, 0.06) # y limit
      plt.xlim(-1.5, 1.5) # x limit
      #plt.plot(x_values, gess, lw = 1, color = 'red', label = '$\phi^{(f)}(x)$' )
      plt.plot(x_values, gess_analitico, lw = 1, color = 'red', label = '$\phi^{(f)}(x)$' )
      plt.plot(x_values, u_zero(x_values), lw = 1, color = 'blue',label = '$\phi^{(t)}(x)$' )
      plt.title(f'Execução {j+1} de {n_testes} utilizando {n_amostras} amostras com $\Delta x = $ {delta_x}.')
      plt.legend()
      plt.pause(0.1)
      # computing the cost
      #Cost1 = n_amostras*functional_cost(u_zero(x_values), gess)
      Cost1 = functional_cost( gess_analitico)
      if Cost1 > Cost:
          print(f'Iteração{j+1}:O funcional custo almentou para = {Cost1}.')
      elif Cost1 < Cost:
          print(f'Iteração{j+1}:O funcional custo diminuiu para = {Cost1}.')
      else:
          print(f'Iteração{j+1}:O funcional custo não alterou')
      error += [Cost1]
      #OrdemDeConvergencia += [convergencia(gess,u_zero(x_values))]
      OrdemDeConvergencia += [convergencia(gess_analitico,u_zero(x_values))]
      Cost = Cost1
      # computing the discrepance
      for i in range(n_amostras):
          d[:,i] = gess - matriz_de_amostras[:,i]
      # computing the gradient
      grad = d[:,i]
      for i in range(n_amostras-1, 0, -1):
          grad = Lax_Friedrichs_transpose(grad)+d[:,i-1]
      # update the inition condition
      gess = gess-0.1*grad
      gess_analitico = gess_analitico + 0.1*(n_amostras/a)*(u_zero(x_values)-gess_analitico)
  plt.show()
  #plt.clf()
  plt.yscale('log')
  plt.xscale('log')
  plt.scatter([i for i in range(n_testes)], OrdemDeConvergencia, lw = 1, color = 'blue',label = '$\mathcal{J}(\phi^{(f)}(x))$ em cada iteração' )
  plt.legend()
  plt.show()
  

else:
  #print('opção inválida.')
  print('End of process.')


