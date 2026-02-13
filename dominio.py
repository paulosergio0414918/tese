import numpy as np

class Dominio:
  """  
  Esta classe tem a finalidade de gerar o domínio a ser discretizado.
  Construirá um eixo x, um eixo y e uma malha que serão usados tanto na
  produção de graficos quanto na execução dos métodos numéricos aqui empregados
  """
  def __init__(self,
                t0: float = 0.0,
                T: float = 2.0,
                L: float = 4.0,
                N: int = 4096,
                M: int = 1024,
                dt: float = None,
                dx: float = None,
                ):
     #Valores default usados em Wicker & Skamarock 2002

    self.t0 = t0   #Tempo inicial (segundos)
    self.T = T     #Tempo final (segundos)
    self.L = L     #Tamanho do intervalo (metros)
    self.N = N     #Número de passos no espaço
    self.M = M     #Número de passos no tempo

    #Discretização no tempo
    if self.M == None:
      self.M = (T-t0)/dt
      if not self.M.is_integer():
        print("Cuidado! Com esse h o número de passos de tempo não é inteiro!!")
        #print("  dt usado: ", (T-t0)/(int((T-t0)/h)))
    else:
      self.dt = (T-t0)/self.M
      #print("Calculando h baseado no n dado. h = ", h)

    #self.n = int((T-t0)/dt)                #Número de passos no tempo
    self.t = np.linspace(t0, T, self.M)  #Tempos discretos

    #Discretização no espaço
    if self.N == None:
      self.N = 2*L/dx
      if not self.N.is_integer():
        print("Cuidado! Com esse dx o número de pontos no espaço não é inteiro!!")
        #print("  dx usado:", (b-a)/(int((b-a)/dx)))
    else:
      self.dx = 2*L/self.N
      #print("Calculando dx baseado no m dado. dx = ", dx)

    #self.m = int(2*L/dx)                #Número de pontos no espaço
    self.x = np.linspace(-L, L, self.N)  #Pontos no espaço

    #self.h = (T-t0)/self.M     #Passo de tempo (segundos)
    #self.dx = 2*L/self.N   #Intervalo espacial (metros)
    cfl = (self.T*self.N)/(2*self.L*self.M)

    if cfl != 1:
      print(f"""Para este domínio temos o número de Courant = {cfl:.2f} tornando o 
            método numérico instável. Recomendamos a seguinte escolha:
            T = 2.0,
            L = 4.0,
            N = 4096,
            M = 1024,
            """)

if __name__ == "__main__":
  dom = Dominio(T=3)
  
 