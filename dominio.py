import numpy as np
from rich.traceback import install
install()


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
    self.validacao_cfl()

    #Discretização no tempo
    if self.M == None:
      self.M = (T-t0)/dt
      if not self.M.is_integer():
        print("Cuidado! Com esse h o número de passos de tempo não é inteiro!!")
        
    else:
      self.dt = (T-t0)/self.M
    
    self.t = np.linspace(t0, T, self.M)  #Tempos discretos

    #Discretização no espaço
    if self.N == None:
      self.N = 2*L/dx
      if not self.N.is_integer():
        print("Cuidado! Com esse dx o número de pontos no espaço não é inteiro!!")

    else:
      self.dx = 2*L/self.N

    self.x = np.linspace(-L, L, self.N)  #Pontos no espaço


    # discretização com malha descentralizada
    self.x_centro = np.array([-self.L+i*self.dx for i in range(self.N)])
    self.x_borda = np.array([-self.L+(i+0.5)*self.dx for i in range(self.N)])

    self.cfl = (self.T*self.N)/(2*self.L*self.M)


  def calculo_cfl(self,
                  n: int = None,
                  m: int = None):
      if n is None:
        n = self.N
      
      if m is None:
        m = self.M

      lam = ((self.T-self.t0)*n)/(2*self.L*m)
      if lam >1 or lam<0:
        return "Instável"
      else:
        return lam
    
  def valores_cfl(self):
    from rich.table import Table
    from rich import print
    passo = 1024
    tab = Table(title = " Sugestões de número de Courant. ")
    tab.add_column(f" ", justify = "center")
    tab.add_column(f"N = {int(passo/(2**2))}", justify = "center")
    tab.add_column(f"N = {int(passo/(2**1))}", justify = "center")
    tab.add_column(f"N = {passo} ", justify = "center")
    tab.add_column(f"N = {passo*(2**1)}", justify = "center")
    tab.add_column(f"N = {passo*(2**2)}", justify = "center")

    for j in range(10):
        tab.add_row(f"M = {2**(j+5)}",
                    f"{self.calculo_cfl(passo/(2**2),2**(j+5))}",
                    f"{self.calculo_cfl(passo/(2**1), 2**(j+5))}",
                    f"{self.calculo_cfl(passo, 2**(j+5))}", 
                    f"{self.calculo_cfl(passo*(2**1), 2**(j+5))}", 
                    f"{self.calculo_cfl(passo*(2**2), 2**(j+5))}"
                    )
    print(tab)
  
  def validacao_cfl(self):
    if self.calculo_cfl() is str:
      print(f"""Para este domínio temos o número de Courant = {self.cfl:.2f} tornando o 
          método numérico instável para Advecção. Recomendamos a seguinte escolha 
          T = 2.0, L = 4.0, N = 4096 e M = 1024.
          
          Para águas rasas sugerimos:
          """
          )
      self.valores_cfl()

    


      



if __name__ == "__main__":
  dom = Dominio(M=512, N=1024)
  print(f"dx = {dom.dx}")
  print(f"dt = {dom.dt}")
  

  
 