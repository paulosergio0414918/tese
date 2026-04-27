import numpy as np
from rich.traceback import install
import random
import matplotlib.pyplot as plt
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
                 c: int = 1, #velocidade de adveccao
                ):
        
        self.condicao = condicao
        self.dom = dom
        #self.iteracoes = iteracoes
        self.c = c 

    def CFL(self):
        """Calcula o número CFL (Condição de Courant-Friedrichs-Lewy)"""
        lam = self.c*self.dom.dt/self.dom.dx
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
    
    def solucao_numerica(self,
                         iteracao: int = None
                         ):
        """Calcula a solução da advecção após vários instantes."""
        if iteracao is None:
            iteracao = self.dom.M
        
        solucao = self.u_zero(self.dom.x) 

        for k in range(iteracao):
            solucao = self.Lax_Friedrichs(solucao)

        return solucao
    
    def solucao_analitica(self,
                          iteracao: int = None
                          ):
        """Fornece a solução analítica do problema de advecção"""
        if iteracao is None:
            iteracao = self.dom.M
        

        return self.u_zero(self.dom.x - self.c*(iteracao+1)*self.dom.dt)
         

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
                vetor_erro += [max(np.abs(s.solucao_analitica()-s.solucao_numerica()))]
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
                axs[i, 0].plot(domi.x, s.solucao_analitica(), label="Sol analítica", linewidth=1)
                axs[i, 0].set_xlabel("x")
                axs[i, 0].set_ylabel("u")
                axs[i, 1].set_title(f"Solução numérica da Advecção para N = {4**(i+2)} e M = {4**(i+1)}")
                axs[i, 1].plot(domi.x, s.solucao_numerica(), label="Sol numérica", linewidth=1)
                axs[i, 1].set_xlabel("x")
                axs[i, 1].set_ylabel("u")

            fig.tight_layout()
            plt.show()


class Assimilacao(SolucaoAdveccao):
    """ Classe destinada a coletar as amostras para assimilação."""
    def __init__(self,
                 dom: Dominio, # um domínio criado pela classe Dominio
                 c: float = 1, #velocidade de advecção
                 n_amostras: int = 2,
                 condicao: str = "condicao_paper",
                 ruido: bool = False,
                 modo: str = "numerico"
                 ):  
        self.n_amostras = n_amostras
        self.dom = dom
        self.c = c
        self.condicao = condicao
        self.passos = [int((dom.M*(dom.T-(dom.T/self.n_amostras)*i))/2) for i in range(self.n_amostras)]
        #self.passos = [int(self.dom.M - np.ceil(self.dom.N/(4*self.dom.L))*i) for i in range(self.n_amostras)]
        self.ruido = ruido
        self.sol = SolucaoAdveccao(self.dom)
        self.modo = modo
        self._matriz_com_amostras = None
        self._matriz_com_amostras_ruido = None
        self.vetor_custo = []
        self.vetor_ruido = [random.uniform(0, 0.0005) for i in range(self.dom.N)]
        self.E = np.linalg.norm(self.vetor_ruido)/self.n_amostras
        
        self.matriz_de_amostras_ruido()

    def matriz_de_amostras(self):
        #if self._matriz_com_amostras is None:
        """Gera uma matriz contendo as amostras sem perturbação"""
        matriz = np.zeros((self.dom.N ,self.n_amostras))
        solu = SolucaoAdveccao(self.dom, self.condicao)
        
        for i, passo in enumerate(self.passos):
            
            matriz[:, i] = solu.solucao_analitica(iteracao=passo)
            
        self._matriz_com_amostras = matriz
        return matriz
        
    def matriz_de_amostras_ruido(self):
        """Gera uma matriz contendo as amostras com perturbação """
        #if self._matriz_com_amostras_ruido is None:
        if self._matriz_com_amostras is  None:
            self.matriz_de_amostras()     
        matriz_com_ruido = self._matriz_com_amostras.copy()
        
        for j in range(self.n_amostras):
            matriz_com_ruido[:,j] += self.vetor_ruido
        self._matriz_com_amostras_ruido = matriz_com_ruido
        return matriz_com_ruido

    def Lax_Friedrichs_transpose(self, x):
        """ Solução do problema backward."""
        cfl = self.sol.CFL()
        return 0.5*((1 - cfl)*np.roll(x,1) + (1 + cfl)*np.roll(x,-1))
    
    def _matriz_de_diferencas(self, solucao):
        """ Cria uma matriz contendo a diferença entre as amostras e a solução"""
        diferencas = np.zeros((self.dom.N,self.n_amostras))
        
        for i, passo in enumerate(self.passos):
            solucao_local = solucao.copy()
            for _ in range(passo):
                solucao_local = self.sol.Lax_Friedrichs(solucao_local)
            if self.ruido:
                diferencas[:, i] = solucao_local - self.matriz_de_amostras_ruido()[:,i]
            else:
                diferencas[:, i] = solucao_local - self.matriz_de_amostras()[:,i]
        return diferencas

    def calculo_do_gradiente_numerico(self, solucao):
        """Cálculo do gradiente conforme Kalnay(2002) pg 183."""
        d = self._matriz_de_diferencas(solucao)     
        grad = np.zeros(self.dom.N)
        
        # Soma das diferenças propagadas para trás
        for i, passo in enumerate(self.passos):
            # Propagação adjunta desde o passo i até o início
            dif = d[:, i].copy()
            
            # Aplica operador adjunto para voltar no tempo
            for _ in range(passo):
                dif = self.Lax_Friedrichs_transpose(dif)
            
            grad += dif
        #print(f"norma do gradiente = {np.linalg.norm(grad)}")
        return grad    

    def calculo_do_gradiente_analitico(self,solucao):
        #calculando as diferencas no tempo (x0-T)/c
        if self.ruido:
            grad = np.zeros(self.dom.N)
            amostra_local = self.sol.u_zero(self.dom.x)
            for i in range(self.n_amostras):
                grad += -(1/self.c)*(amostra_local + self.vetor_ruido - solucao)
        else:
            grad = np.zeros(self.dom.N)
            amostra_local = self.sol.u_zero(self.dom.x)
            for _ in range(self.n_amostras):
                grad += -(1/self.c)*(amostra_local - solucao) 

        return grad
        
    def gradiente_descendente(self,
                              it:int = 10):
        """Calculo do gradiente descendente considerando n=it iterações"""
        solucao_final = np.zeros(dom.N) #chute inicial
        if self.modo == "numerico":
            for i in range(it):
                grad = self.calculo_do_gradiente_numerico(solucao = solucao_final)
                solucao_final = solucao_final - 0.1*grad
        else:
            for i in range(it):             
                grad = self.calculo_do_gradiente_analitico(solucao = solucao_final)
                solucao_final = solucao_final - 0.1*grad            

        return solucao_final
     
    def custo_assimilacao(self,
                          it: int = 10,
                          grafico: bool = False
                          ):
        """Retorna o custo de assimilação para cada iteração."""
        
        solucao = np.zeros(self.dom.N)
        
        for i in range(1, it + 1):
            # Atualiza solução
            solucao = self.gradiente_descendente(i)
            
            # Calcula custo
            custo = 0
            for passo in self.passos:
                # Propaga solução
                sol_propagada = solucao.copy()
                for _ in range(passo):
                    sol_propagada = self.sol.Lax_Friedrichs(sol_propagada)
                
                # Diferença com observação
                if self.ruido:
                    matriz_obs = self.matriz_de_amostras_ruido() 
                    obs = matriz_obs[:, self.passos.index(passo)]
                else:
                    matriz_obs = self.matriz_de_amostras()    
                    obs = matriz_obs[:, self.passos.index(passo)]
                diff = sol_propagada - obs
                custo += 0.5 * np.dot(diff, diff)
            
            self.vetor_custo.append(custo)
        if grafico:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_xlabel('Número de iterações')
            #ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel('J(x)')
            ax.scatter([i+1 for i in range(it)], self.vetor_custo, lw = 0.5, color = 'blue',label = '$J(x_0)$ em cada iteração' )
            ax.legend()
            plt.show()
            return fig
        else:
            return self.vetor_custo
    
    def diferenca(self,
                  iter: int = 10):
        diferenca = []
        for i in range(iter):
            diferenca += [np.linalg.norm(self.sol.u_zero(dom.x)-self.gradiente_descendente(it = i))]

        return diferenca


if __name__ == "__main__":
    import dominio
    import construtor_de_graficos as cdg

    ###### parâmetros #######
    op = 12
    ruido = True
    iteracoes = 32
    amos = 5


    ###### objetos ##########
    dom = dominio.Dominio()
    ass = Assimilacao(dom, modo="analitico", n_amostras = amos)
    sol = SolucaoAdveccao(dom)
    val = Validacao()
    
    if op == 12: #Grafico de todas as amostras
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        
        matriz = ass.matriz_de_amostras()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(amos):
            ax.plot(np.full_like(dom.x, i+1), dom.x, matriz[:, i], linewidth=2)

        ax.set_xlabel('Amostra')
        ax.set_ylabel('x')
        ax.set_zlabel('u')
        ax.set_xticks([i for i in range(amos)])
        ax.view_init(25, -60)
        plt.show()

    elif op == 11: # Constatação do resultado teórico para advecção
            k = ass.E
            d1 = ass.diferenca(iter = iteracoes)
            ass2 = Assimilacao(dom, modo="analitico", ruido= True, n_amostras=4)            
            m = ass2.E
            d2 = ass2.diferenca(iter = iteracoes)
            ass3 = Assimilacao(dom, modo="numérico", ruido= True, n_amostras=2)
            n = ass3.E
            d3 = ass3.diferenca(iter = iteracoes)
            ass4 = Assimilacao(dom, modo="numerico", ruido= True, n_amostras=4)
            d4 = ass4.diferenca(iter = iteracoes)
            ass5 = Assimilacao(dom, modo="analitico")
            d5 = ass5.diferenca(iter = iteracoes)
            ass6 = Assimilacao(dom, modo="analitico",  n_amostras=4) 
            d6 = ass6.diferenca(iter = iteracoes)           
            ass7 = Assimilacao(dom, modo="numérico", n_amostras=2)
            d7 = ass7.diferenca(iter = iteracoes)
            ass8 = Assimilacao(dom, modo="numérico", n_amostras=4)
            d8 = ass8.diferenca(iter = iteracoes)


            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].set_title(f"Método analítico com duas amostras com ruido")
            axs[0, 0].scatter([i+1 for i in range(iteracoes)], d1, label="Com ruido", linewidth=1)
            axs[0, 0].scatter([i+1 for i in range(iteracoes)], d5, label="Sem ruido", linewidth=1)
            axs[0, 0].scatter([i+1 for i in range(iteracoes)], [k for _ in range(iteracoes)], label="|E|", linewidth=1)
            axs[0, 0].legend()
            axs[0, 0].set_xlabel("n")
            axs[0, 0].set_ylabel("|\phi^{(t)}(x) - \phi^{(n)}(x)|")
            axs[0, 0].set_yscale("log"),


            axs[0, 1].set_title(f"Método analítico com quatro amostras")
            axs[0, 1].scatter([i+1 for i in range(iteracoes)], d2, label="Com ruido", linewidth=1)
            axs[0, 1].scatter([i+1 for i in range(iteracoes)], d5, label="Sem ruido", linewidth=1)            
            axs[0, 1].scatter([i+1 for i in range(iteracoes)], [m for _ in range(iteracoes)], label="|E|", linewidth=1)
            axs[0, 1].legend()
            axs[0, 1].set_xlabel("n")
            axs[0, 1].set_ylabel("|\phi^{(t)}(x) - \phi^{(n)}(x)|")
            axs[0, 1].set_yscale("log")

            axs[1, 0].set_title(f"Método numérico com duas amostras")
            axs[1, 0].scatter([i+1 for i in range(iteracoes)], d3, label="Com ruido", linewidth=1)
            axs[1, 0].scatter([i+1 for i in range(iteracoes)], d7, label="Sem ruido", linewidth=1)
            axs[1, 0].scatter([i+1 for i in range(iteracoes)], [n for _ in range(iteracoes)], label="|E|", linewidth=1)
            axs[1, 0].legend()
            axs[1, 0].set_xlabel("n")
            axs[1, 0].set_ylabel("|\phi^{(t)}(x) - \phi^{(n)}(x)|")
            axs[1, 0].set_yscale("log")

            axs[1, 1].set_title(f"Método numérico com quatro amostras")
            axs[1, 1].scatter([i+1 for i in range(iteracoes)], d4, label=" Com ruido", linewidth=1)
            axs[1, 1].scatter([i+1 for i in range(iteracoes)], d8, label=" Sem ruido", linewidth=1)
            axs[1, 1].scatter([i+1 for i in range(iteracoes)], [ass4.E for _ in range(iteracoes)], label="|E|", linewidth=1)
            axs[1, 1].legend()
            axs[1, 1].set_xlabel("n")
            axs[1, 1].set_ylabel("|\phi^{(t)}(x) - \phi^{(n)}(x)|")
            axs[1, 1].set_yscale("log")
            fig.tight_layout()



            #fig.legend()
            plt.show()

    elif op == 10: #comparar os dois métodos de assimilação 
        
        
        import math
        from rich import print
        from rich.table import Table
        if ruido:
            tab = Table(title = "Custo de assimilação com ruído nas amostras.")
        else: 
            tab = Table(title = "Custo de assimilação sem ruído nas amostras.")
        tab.add_column("Itecaçoes", justify = "center")
        tab.add_column("analítico com 2 amostras", justify = "center")
        tab.add_column("analítico com 4 amostras", justify = "center")
        tab.add_column("numerico com 2 amostras", justify = "center")
        tab.add_column("numerico com 4 amostras", justify = "center")
        
                
        for i in range(iteracoes):
            ass1 = Assimilacao(dom, modo = "analitico", ruido= ruido)
            custo1 = min(ass1.custo_assimilacao(it = i+1))
            ass2 = Assimilacao(dom, modo = "analitico", ruido= ruido, n_amostras= 4)
            custo2 = min(ass2.custo_assimilacao(it = i+1))
            ass3 = Assimilacao(dom, modo = "numerico", ruido= ruido)
            custo3 = min(ass3.custo_assimilacao(it = i+1))
            ass4 = Assimilacao(dom, modo = "numerico", ruido= ruido, n_amostras= 4)
            custo4 = min(ass4.custo_assimilacao(it = i+1))
            
            
            j = math.log2(i+1)
            if j.is_integer() or i == 0:
                tab.add_row(f"{i+1}", f"{custo1:.4e}", f"{custo2:.4e}", f"{custo3:.4e}", f"{custo4:.4e}" )

        print(tab)      

    elif op == 9: # gráfico da condição inicial assimilada
        for i in range(10):
            plt.clf()
            y = ass.gradiente_descendente(it=i)
            z = ass.u_zero(dom.x)
            graf = cdg.Grafico2d(dom.x, y1=y,y2=z, y1_name="assimilação", y2_name="realidade", title=f"iteração{i}")
            graf.plot2d()

    elif op == 8: # gráfico teste das amostras
        y = ass.matriz_de_amostras_ruido()[:, 0]
        z = ass.matriz_de_amostras_ruido()[:, 1]
        graf = cdg.Grafico2d(dom.x, y, z,
                             y1_name="primeira amostra",
                             y2_name="segunda amostra" ,
                             title = "As duas soluções")
        graf.plot2d()  

    elif op == 7: #gráficos de validação do método numérico
        val.graficos()

    elif op == 6: # tabela de validação do método numérico
        val.tabela()

    elif op == 5: #teste de assimilação
        y = ass.gradiente_descendente(it=iteracoes)
        graf = cdg.Grafico2d(dom.x, y)
        graf.plot2d()
    
    elif op == 4: #teste do custo de assimilação
        ass.custo_assimilacao(it = iteracoes, grafico=True)
    
    elif op == 3: # teste da solução analítica
        y = sol.solucao_analitica()
        graf = cdg.Grafico2d(dom.x, y, title = "Solução analítica")
        graf.plot2d()        

    elif op == 2: # teste da solução numérica
        y = sol.solucao_numerica()
        graf = cdg.Grafico2d(dom.x, y, title = "Solução numérica")
        graf.plot2d()

    elif op == 1: # diferenção entre as duas soluções
        y = sol.solucao_analitica()
        z = sol.solucao_numerica()
        print(max(y-z))

    elif op == 0: # grafico das duas soluções 
        y = sol.solucao_numerica()
        z = sol.solucao_analitica()
        graf = cdg.Grafico2d(dom.x, y, z, title = "As duas soluções")
        graf.plot2d()
    
