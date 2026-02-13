import numpy as np
from rich.traceback import install
import random
install()

import condicoes_iniciais as ci
from dominio import Dominio


class SolucaoAdveccao:
    """
    ============================================================================
    SOLUÇÃO NUMÉRICA E ANALÍTICA DA EQUAÇÃO DE ADVECÇÃO 1D
    ============================================================================
    
    OBJETIVO:
    -----------
    Discretizar e resolver a equação de advecção linear 1D: 
    
        ∂u/∂t + a ∂u/∂x = 0
    
    utilizando o método de Lax-Friedrichs. A classe também fornece a solução 
    analítica para comparação e coleta amostras para assimilação de dados.
    
    DEPENDÊNCIAS:
    ---------------
    - numpy (np): operações matriciais e rolagem de arrays
    - Dominio: classe externa que define o grid espacial e temporal
    - ci.Funcoes2d: módulo com condições iniciais predefinidas
    
    ============================================================================
    INICIALIZAÇÃO
    ============================================================================
    
    Parâmetros:
    -----------
    dom : Dominio
        Objeto da classe Dominio contendo a discretização do problema.
        Deve possuir os seguintes atributos:
        - dom.dt : passo de tempo
        - dom.dx : passo espacial
        - dom.N  : número de pontos na malha espacial
        - dom.M  : número de pontos na malha temporal
        - dom.x  : array com as coordenadas espaciais
    
    condicao : str, default="condicao_paper"
        Tipo de condição inicial a ser utilizada:
        - "condicao_paper" : condição inicial do artigo de referência
        - "condicao_caixa" : função caixa (degrau)
        - (outras strings) : fallback para "condicao_paper"
    
    a : int, default=1
        Velocidade de advecção (constante positiva).
        Valores típicos: 0.5, 1.0, 2.0
    
    Observação:
    -----------
        O parâmetro 'iteracoes' (comentado) pode ser reativado para controlar
        o número de iterações em versões futuras.
    
    ============================================================================
    ATRIBUTOS
    ============================================================================
    
    self.dom : Dominio
        Objeto com a discretização do problema (ver parâmetros acima).
    
    self.condicao : str
        Tipo de condição inicial selecionada.
    
    self.a : int
        Velocidade de advecção.
    
    ============================================================================
    MÉTODOS PRINCIPAIS
    ============================================================================
    
    1. CFL()
    --------
    Calcula o número de Courant-Friedrichs-Lewy (CFL):
        λ = a * Δt / Δx
    
    Retorno:
        float : Número CFL (deve ser ≤ 1 para estabilidade)
    
    Exemplo:
        sol = SolucaoAdveccao(dominio, a=1)
        cfl = sol.CFL()
        print(f"CFL = {cfl:.3f}")
    
    --------
    2. u_zero(x)
    --------
    Define a condição inicial u(x, t=0).
    
    Parâmetros:
        x : float ou np.ndarray
            Coordenada(s) espacial(is) para avaliação
    
    Retorno:
        float ou np.ndarray : Valor(es) da condição inicial
    
    Comportamento:
        - Se condicao == "condicao_caixa": retorna função caixa
        - Caso contrário: retorna condição do paper (padrão)
    
    --------
    3. Lax_Friedrichs(u)
    --------
    Aplica um passo do esquema de Lax-Friedrichs.
    
    Parâmetros:
        u : tuple ou np.ndarray
            Solução no instante atual t_n (array 1D)
    
    Retorno:
        np.ndarray : Solução no instante t_{n+1}
    
    Esquema numérico:
        u_j^{n+1} = 0.5[(1+λ)u_{j-1}^n + (1-λ)u_{j+1}^n]
    
    --------
    4. solucao_numerica()
    --------
    Integra a equação ao longo do tempo usando Lax-Friedrichs.
    
    Retorno:
        np.ndarray : Matriz (N x M) com a solução numérica completa
                    - Linhas: posições espaciais
                    - Colunas: instantes temporais
    
    Algoritmo:
        1. Inicializa matriz de solução com zeros
        2. Aplica condição inicial na primeira coluna
        3. Para cada passo temporal: aplica Lax_Friedrichs
    
    --------
    5. solucao_analitica()
    --------
    Calcula a solução analítica exata da equação de advecção.
    
    Retorno:
        np.ndarray : Matriz (N x M) com a solução analítica
    
    Fórmula:
        u(x, t) = u_zero(x - a·t)
        onde u_zero é a condição inicial
    
    ============================================================================
    EXEMPLO COMPLETO
    ============================================================================
    
    import numpy as np
    from dominio import Dominio
    from condicoes_iniciais import Funcoes2d as ci
     
    # 1. Criar domínio
    dom = Dominio(L=100, N=201, T=50, M=101)
    
    # 2. Instanciar solver
    solver = SolucaoAdveccao(dom, condicao="condicao_caixa", a=1.5)
     
    # 3. Verificar estabilidade
    cfl = solver.CFL()
    print(f"CFL = {cfl:.3f}")
    if cfl > 1:
        print(" ATENÇÃO: CFL > 1 - instabilidade numérica!")
        print(" Reduza Δt ou aumente Δx.")
    
    # 4. Calcular soluções
    u_numerica = solver.solucao_numerica()
    u_analitica = solver.solucao_analitica()
    
    # 5. Calcular erro
    erro = np.max(np.abs(u_numerica - u_analitica))
    print(f"Erro máximo = {erro:.2e}")
    """

    def __init__(self, 
                 dom: Dominio, # um domínio criado pela classe Dominio 
                 condicao: str = "condicao_paper", #condicao inicial para o problema
                 #teracoes: int = 100, # numero de iterações do metodo de solucao
                 a: int = 1, #velocidade de adveccao
                ):
        
        self.condicao = condicao
        self.dom = dom
        #self.iteracoes = iteracoes
        self.a = a 


    def CFL(self):
        """Calcula o número CFL (Condição de Courant-Friedrichs-Lewy)"""
        lam = self.a*self.dom.dt/self.dom.dx
        #print(f"Número de Courant = {lam}")
        return lam
        #return (2*self.a*dom.L*dom.M)/(dom.N*dom.T)


    def u_zero(self, x):
        """Define a condição inicial no eixo u"""
        condicao = ci.Funcoes2d()

        if self.condicao == "condicao_caixa":
            return condicao.condicao_caixa(x)
        
        else:
            return condicao.condicao_paper(x)

    
    def Lax_Friedrichs(self,
                       u: tuple ):
        """Resolução da equação da advecção """
        cfl = self.CFL()
        return 0.5*((1 + cfl)*np.roll(u,1) + (1 - cfl)*np.roll(u,-1))
    
    def solucao_numerica(self):
        """Calcula a solução da advecção após vários instantes."""
        solucao = np.zeros((self.dom.N, self.dom.M))
        solucao[:,0] = self.u_zero(self.dom.x) 

        for k in range(self.dom.M-1):
            solucao[:,k+1] = self.Lax_Friedrichs(solucao[:,k])

        return solucao
    
    def solucao_analitica(self):
        """Fornece a solução analítica do problema de advecção"""
        solucao = np.zeros((self.dom.N, self.dom.M))
        for j in range(self.dom.M):
            solucao[:,j] = self.u_zero(self.dom.x-self.a*(j+1)*self.dom.dt)
        return solucao

class Validacao(SolucaoAdveccao):
        """
        ===========================================================================
        VALIDAÇÃO E ANÁLISE DE CONVERGÊNCIA DO MÉTODO DE LAX-FRIEDRICHS
        ============================================================================
        
        OBJETIVO:
        -----------
        Esta classe herda de SolucaoAdveccao e implementa ferramentas para validar
        a solução numérica da equação de advecção, analisando:
        
        - Ordem de convergência do método numérico
        - Erros de aproximação em diferentes refinamentos de malha
        - Comparação visual entre soluções analítica e numérica
        - Condição de estabilidade CFL

        ============================================================================
        ATRIBUTOS
        ============================================================================
        
        self.testes : int
            Número de refinamentos a serem processados.
        
        (Os demais atributos são herdados de SolucaoAdveccao quando
        uma instância de Validacao cria objetos SolucaoAdveccao internos)
        
        ============================================================================
        MÉTODOS
        ============================================================================
        
        1. tabela()
        -----------
        Gera e exibe uma tabela formatada com análise de convergência.

        
        2. graficos()
        -----------
        Gera subplots comparativos entre soluções analítica e numérica.
        """

        def __init__(self,
                     testes: int = 6):
            self.testes = testes
            #import dominio


        def tabela(self):
            """ Apresenta uma tabela com os erros de aproximação """
            import math
            from rich import print
            from rich.table import Table
            vetor_erro = []
            tab = Table(title = "Ordem de convergência")
            tab.add_column("N", justify = "center")
            tab.add_column("M", justify = "center")
            tab.add_column("Courant", justify = "center")
            tab.add_column("Erro", justify = "center")
            tab.add_column("Ordem", justify = "center", style = "red")

            for j in range(self.testes):
                domi = dominio.Dominio(N = 4**(j+2),  M = 4**(j+1))
                s = SolucaoAdveccao(domi)
                vetor_erro += [max(np.abs(s.solucao_analitica()[:,-1]-s.solucao_numerica()[:,-1]))]
                if j == 0:
                    tab.add_row(f"{domi.N}", f"{domi.M}", f"{s.CFL()}", f"{vetor_erro[j]:.4e}", None )
                else:
                    tab.add_row(f"{domi.N}", f"{domi.M}", f"{s.CFL()}", f"{vetor_erro[j]:.4e}", f"{math.log(abs(vetor_erro[j-1]/vetor_erro[j]))/math.log(4):.4e}" )
    
            print(tab)

        def graficos(self):
            """Apresenta o grafico para da solução analítica e numérica variando os valores de M e de N"""
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(self.testes, 2)
            for i in range(self.testes):

                domi = dominio.Dominio(N = 4**(i+2),  M = 4**(i+1))
                s = SolucaoAdveccao(domi)
                axs[i, 0].set_title(f"Solução analítica da Advecção para N = {4**(i+2)} e M = {4**(i+1)}")
                axs[i, 0].plot(domi.x, s.solucao_analitica()[:,-1], label="Sol analítica", linewidth=1)
                axs[i, 0].set_xlabel("x")
                axs[i, 0].set_ylabel("u")
                axs[i, 1].set_title(f"Solução numérica da Advecção para N = {4**(i+2)} e M = {4**(i+1)}")
                axs[i, 1].plot(domi.x, s.solucao_numerica()[:,-1], label="Sol numérica", linewidth=1)
                axs[i, 1].set_xlabel("x")
                axs[i, 1].set_ylabel("u")

            fig.tight_layout()
            plt.show()


class Assimilacao(SolucaoAdveccao):
    """ Classe destinada a coletar as amostras para assimilação."""
    def __init__(self,
                 dom: Dominio, # um domínio criado pela classe Dominio 
                 n_amostras: int = 2,
                 condicao: str = "condicao_paper",
                 ruido: bool = False
                 ):  
        self.n_amostras = n_amostras
        self.dom = dom
        self.condicao = condicao
        self.passos = np.flip(-1 + np.linspace(dom.M, 0 , self.n_amostras, dtype = int, endpoint=False))
        self.ruido = ruido
        self.sol = SolucaoAdveccao(self.dom)


    def matriz_de_amostras(self):
        """Gera uma matriz contendo as amostras sem perturbação"""
        matriz = np.zeros((self.dom.N ,self.n_amostras))
        
        for i in range(self.n_amostras):
            solu = SolucaoAdveccao(self.dom, self.condicao)
            matriz[:, i] = solu.solucao_analitica()[: , int(self.passos[i])]
            
        return matriz
    
    def matriz_de_amostras_ruido(self):
        """Gera uma matriz contendo as amostras com perturbação """
        matriz_com_ruido = self.matriz_de_amostras()
        for i in range(self.dom.N):
            for j in range(self.n_amostras):
                matriz_com_ruido[i,j] += random.uniform(0, 0.0005)
        return matriz_com_ruido

    def Lax_Friedrichs_transpose(self, x):
        """ Solução do problema backward."""
        cfl = self.sol.CFL()
        return 0.5*((1 - cfl)*np.roll(x,1) + (1 + cfl)*np.roll(x,-1))
    
    def __matriz_de_diferencas__(self, solucao):
        diferencas = np.zeros((dom.N,self.n_amostras))
        
        for i in range(self.n_amostras):
            for _ in range(self.passos[i]):
                solucao = self.sol.Lax_Friedrichs(solucao)
            if self.ruido:
                diferencas[:, i] = solucao - self.matriz_de_amostras_ruido()[:,i]
            else:
                diferencas[:, i] = solucao - self.matriz_de_amostras()[:,i]
        return diferencas

    def calculo_do_gradiente(self, solucao):
        d = self.__matriz_de_diferencas__(solucao) 
        grad = d[:, self.n_amostras-1] 
        for i in range(self.n_amostras-1, 0, -1):
            for _ in range(self.passos[i]-self.passos[i-1]):
                grad =  self.Lax_Friedrichs_transpose(grad)    
            grad += d[:, i-1]
        #print(grad.shape)    
        return grad

    def gradiente_descendente(self,
                              it:int = 10):
        solucao_final = np.zeros(dom.N)
        for _ in range(it):
            grad = self.calculo_do_gradiente(solucao = solucao_final)
            solucao_final = solucao_final - 0.1*grad
        return solucao_final 

    def custo_assimilacao(self,
                          it:int = 10):
        vetor_custo = []
        cost = 0
        for _ in range(it):
            ass = self.gradiente_descendente(it+1)
            for i in range(self.passos[-1]):
                self.sol.Lax_Friedrichs(ass)
                if i in self.passos:
                   v = ass - self.matriz_de_amostras()[:,np.where(self.passos == i)[0][0]]
                   cost += v @ v
            vetor_custo += [cost]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Número de iterações')
        ax.set_ylabel('J(x)')
        ax.scatter([i+1 for i in range(it)], vetor_custo, lw = 0.5, color = 'blue',label = '$J(x_0)$ em cada iteração' )
        ax.legend()
        plt.show()
        return fig
         



        #1º resolver o problema direto com um chute calculando as diferenças
        #2º armazenar todas as diferenças
        #3º usar o método transposto para obter o gradiente   
    #def assimilacao_numerica(self):
    #    pass
        
            

if __name__ == "__main__":
    import dominio
    import construtor_de_graficos as cdg
    #criando o objeto amostras
    dom = dominio.Dominio()
    ass = Assimilacao(dom)
    y = ass.gradiente_descendente()
    graf = cdg.Grafico2d(dom.x, y)
    graf.plot2d()
