import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from rich.traceback import install
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
        self.ruido = ruido
        self.sol = SolucaoAdveccao(self.dom)
        self.modo = modo
        self._matriz_com_amostras = None
        self._matriz_com_amostras_ruido = None
        self.vetor_custo = []
        self.vetor_ruido = [random.uniform(0, 0.005) for i in range(self.dom.N)]
        self.E = np.linalg.norm(self.vetor_ruido)/self.n_amostras
        self.tj = [((dom.M*(dom.T-(dom.T/self.n_amostras)*i))/2)*dom.dt for i in range(self.n_amostras)]

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

    
    # aproximação do ruido branco segundo 
    #Allen, E. J., Novosel, S. J., & Zhang, Z. (1998). Finite element and difference approximation of some linear stochastic partial differential equations. Stochastics and Stochastic Reports, 64(1–2), 117–142. https://doi.org/10.1080/17442509808834159

    def white_noise(self,
                    scale: float = 0.1
                    ):

        return scale*max(self.sol.solucao_analitica())*np.sqrt(self.dom.dx)*np.random.randn(self.dom.N)
            
    def bronwniano(self,
                   t: float = 0,
                   seed: int = 100
                   ):
        
        np.random.seed(seed) 
        incrementos = np.sqrt(dom.T-t)*np.sqrt(dom.T/dom.M)*np.random.randn(dom.N)        
        W = np.zeros(dom.N)
        W[0] = incrementos[0]
        for i in range(1, dom.N):
            W[i] = W[i-1] + incrementos[i]  

        return W
    
    def matriz_b(self):
        matrizb = np.zeros((dom.N, self.n_amostras))
        for j in range(self.n_amostras):
            matrizb[:, j] = self.bronwniano(t = self.tj[j])

        return matrizb
    
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

#preciso inserir um método para diferenças considerando a solução analítica

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
    
    def calculo_do_gradiente_estocastico(self, solucao):
        grad = np.zeros(self.dom.N)
        amostra_local = self.sol.u_zero(self.dom.x)
        mean = np.mean(self.matriz_b(), axis=1)
        for i in range(self.n_amostras):
            grad += -(1/self.c)*(amostra_local - solucao - (1/self.n_amostras)*mean)

        return grad

    def gradiente_descendente_otimizado(self, it: int = 10):
        """Gradiente descendente com busca de linha (passo ótimo a cada iteração)."""
        solucao_final = np.zeros(self.dom.N)  # chute inicial

        for _ in range(it):
            # Calcula o gradiente conforme o modo escolhido
            if self.modo == "numerico":
                grad = self.calculo_do_gradiente_numerico(solucao=solucao_final)
            elif self.modo == "estocastico":
                grad = self.calculo_do_gradiente_estocastico(solucao=solucao_final)
            else:  # analitico
                grad = self.calculo_do_gradiente_analitico(solucao=solucao_final)

            # Função que calcula o custo para um dado passo alpha
            def custo_alpha(alpha):
                return self.custo_solucao(solucao_final - alpha * grad)

            # Encontra o alpha ótimo no intervalo [0, 1] (ajuste se necessário)
            res = minimize_scalar(custo_alpha, bounds=(0, 1), method='bounded',  options={'maxiter': 5})
            alpha_otimo = res.x

            # Atualiza a solução com o passo encontrado
            solucao_final = solucao_final - alpha_otimo * grad

        return solucao_final
    
    def gradiente_descendente(self,
                              it:int = 10):
        """Calculo do gradiente descendente considerando n=it iterações"""
        solucao_final = np.zeros(dom.N) #chute inicial
        if self.modo == "numerico":
            for i in range(it):
                grad = self.calculo_do_gradiente_numerico(solucao = solucao_final)
                solucao_final = solucao_final - 0.1*grad
        elif self.modo == "estocastico":
            for i in range(it):
                grad = self.calculo_do_gradiente_estocastico(solucao = solucao_final)
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
            diferenca += [np.mean(self.sol.u_zero(dom.x)-self.gradiente_descendente(it = i))]

        return diferenca
    
    def custo_solucao(self, solucao):
        """Calcula o custo J(x) para uma dada solução (sem gradiente)."""
        custo = 0.0
        for i, passo in enumerate(self.passos):
            sol_propagada = solucao.copy()
            for _ in range(passo):
                sol_propagada = self.sol.Lax_Friedrichs(sol_propagada)
            if self.ruido:
                obs = self._matriz_com_amostras_ruido[:, i]
            else:
                obs = self._matriz_com_amostras[:, i]
            diff = sol_propagada - obs
            custo += 0.5 * np.dot(diff, diff)
        return custo



if __name__ == "__main__":
    import dominio
    import construtor_de_graficos as cdg
    import matplotlib.pyplot as plt

    ###### parâmetros #######
    op = 19
    ruido = True
    iteracoes = 16
    amos = 2


    ###### objetos ##########
    dom = dominio.Dominio()
    ass = Assimilacao(dom, modo="analitico", n_amostras = amos)
    sol = SolucaoAdveccao(dom)
    val = Validacao()
    
    ##### lista de testes ##########
    if op == 19: # Grafico das amostras em 3d
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.animation import FuncAnimation

        # Definindo a função
        def u(x, t):
            return (1/20) * np.exp(-10 * (x - t)**2)

        # ============================================
        # PARÂMETROS
        # ============================================
        x_const = 1.0
        t_min, t_max = 0, 2

        # Domínios para o gráfico 3D
        x = np.linspace(-1, 3, 100)
        t = np.linspace(t_min, t_max, 100)

        # Criando a malha
        X, T = np.meshgrid(x, t)
        U = u(X, T)

        # Dados para a curva x = 1
        t_curve = t
        u_curve = u(x_const, t_curve)
        x_const_array = np.full_like(t_curve, x_const)

        # Dados para a animação 2D (mais pontos para suavidade)
        t_anim = np.linspace(t_min, t_max, 300)
        u_at_x1 = u(x_const, t_anim)

        # Ponto de máximo (para referência)
        t_pico = x_const
        u_pico = u(x_const, t_pico)

        # ============================================
        # CRIAÇÃO DA FIGURA COM DOIS SUBPLOTS
        # ============================================
        fig = plt.figure(figsize=(16, 8))

        # Subplot 1: Gráfico 3D (esquerda)
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')

        # Subplot 2: Animação 2D (direita)
        ax2d = fig.add_subplot(1, 2, 2)

        # ============================================
        # GRÁFICO 3D
        # ============================================
        # Plotando a superfície
        surf = ax3d.plot_surface(X, T, U, cmap='viridis', alpha=0.7, 
                                linewidth=0, antialiased=True)

        # Plotando a curva para x = 1 constante
        curve_line, = ax3d.plot(x_const_array, t_curve, u_curve, 
                                color='red', linewidth=2.5, 
                                label=f'x = {x_const}')

        # Ponto animado no gráfico 3D (vai acompanhar a animação 2D)
        point_3d, = ax3d.plot([], [], [], 'ro', markersize=8, 
                            markeredgecolor='black', 
                            markeredgewidth=1.5, zorder=10)

        # Linha vertical conectando o ponto ao plano base
        vertical_line, = ax3d.plot([], [], [], 'gray', linewidth=1, 
                                alpha=0.5, linestyle='--')

        # Configurando o gráfico 3D
        ax3d.set_xlabel('x (Posição)', fontsize=10, labelpad=8)
        ax3d.set_ylabel('t (Tempo)', fontsize=10, labelpad=8)
        ax3d.set_zlabel('u(x,t)', fontsize=10, labelpad=8)
        ax3d.set_title('Superfície 3D: u(x,t) = (1/20)·e^(-10·(x-t)²)\n'
                    f'Curva para x = {x_const} (vermelha)', 
                    fontsize=10, pad=15)

        ax3d.set_xlim(-1, 3)
        ax3d.set_ylim(t_min, t_max)
        ax3d.set_zlim(0, 0.052)
        ax3d.view_init(elev=25, azim=55)
        ax3d.grid(True, alpha=0.2)
        ax3d.legend(loc='upper left', fontsize=8)

        # Barra de cores
        cbar = fig.colorbar(surf, ax=ax3d, shrink=0.6, aspect=15, pad=0.1)
        cbar.set_label('Amplitude', fontsize=9)

        # ============================================
        # GRÁFICO 2D (ANIMAÇÃO)
        # ============================================
        # Curva estática de u(1,t)
        ax2d.plot(t_anim, u_at_x1, 'b-', linewidth=2, alpha=0.6, 
                label=f'u({x_const}, t)')

        # Preenchimento sob a curva
        ax2d.fill_between(t_anim, 0, u_at_x1, alpha=0.2, color='blue')

        # Ponto animado no gráfico 2D
        point_2d, = ax2d.plot([], [], 'ro', markersize=10, 
                            markeredgecolor='black', 
                            markeredgewidth=1.5, zorder=5)

        # Linha vertical animada no gráfico 2D
        vertical_line_2d, = ax2d.plot([], [], 'r--', linewidth=1.5, alpha=0.5)

        # Linha de referência do pico
        ax2d.axvline(x=t_pico, color='red', linestyle='--', alpha=0.4, 
                    linewidth=1, label=f'Pico em t = {t_pico}')

        # Ponto estático do pico (referência)
        ax2d.scatter([t_pico], [u_pico], color='darkred', s=50, 
                    edgecolor='black', linewidth=1, alpha=0.5, zorder=3)

        # Textos informativos
        time_text = ax2d.text(0.02, 0.95, '', transform=ax2d.transAxes, 
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', 
                                    alpha=0.8, edgecolor='orange'))

        value_text = ax2d.text(0.02, 0.85, '', transform=ax2d.transAxes, 
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightblue', 
                                        alpha=0.8, edgecolor='blue'))

        # Configurando o gráfico 2D
        ax2d.set_xlabel('Tempo (t)', fontsize=11)
        ax2d.set_ylabel(f'u({x_const}, t) - Amplitude', fontsize=11)
        ax2d.set_title(f'Evolução Temporal no Ponto x = {x_const}\n'
                    f'Animação: ponto se movendo ao longo da curva', 
                    fontsize=10, pad=15)
        ax2d.set_xlim(t_min, t_max)
        ax2d.set_ylim(0, 0.052)
        ax2d.grid(True, alpha=0.3, linestyle='--')
        ax2d.legend(loc='upper right', fontsize=8)
        ax2d.set_facecolor('#f8f9fa')
        ax2d.axhline(y=0, color='black', linewidth=0.5)

        # ============================================
        # FUNÇÃO DE ANIMAÇÃO (ATUALIZA AMBOS OS GRÁFICOS)
        # ============================================
        def animate(frame):
            """
            Atualiza a animação tanto no gráfico 3D quanto no 2D
            frame: índice do frame atual (0 a 299)
            """
            # Tempo atual baseado no frame
            t_current = t_anim[frame]
            
            # Calcula o valor de u no ponto (x=1, t_current)
            u_current = u(x_const, t_current)
            
            # ===== ATUALIZA GRÁFICO 3D =====
            # Atualiza posição do ponto 3D
            point_3d.set_data([x_const], [t_current])
            point_3d.set_3d_properties([u_current])
            
            # Atualiza linha vertical no 3D
            vertical_line.set_data([x_const, x_const], [t_current, t_current])
            vertical_line.set_3d_properties([0, u_current])
            
            # ===== ATUALIZA GRÁFICO 2D =====
            # Atualiza posição do ponto 2D
            point_2d.set_data([t_current], [u_current])
            
            # Atualiza linha vertical no 2D
            vertical_line_2d.set_data([t_current, t_current], [0, u_current])
            
            # Atualiza textos
            time_text.set_text(f'tempo = {t_current:.3f} s')
            value_text.set_text(f'u = {u_current:.5f}')
            
            # ===== EFEITOS VISUAIS =====
            # Muda a cor do ponto baseado na amplitude (ambos os gráficos)
            intensidade = u_current / u_pico
            
            if intensidade > 0.95:
                cor = 'darkred'
                size_3d = 10
                size_2d = 12
            elif intensidade > 0.7:
                cor = 'red'
                size_3d = 9
                size_2d = 11
            elif intensidade > 0.3:
                cor = 'orange'
                size_3d = 8
                size_2d = 10
            else:
                cor = 'yellow'
                size_3d = 7
                size_2d = 9
            
            point_3d.set_color(cor)
            point_3d.set_markersize(size_3d)
            point_2d.set_color(cor)
            point_2d.set_markersize(size_2d)
            
            return point_3d, vertical_line, point_2d, vertical_line_2d, time_text, value_text

        # ============================================
        # CRIAÇÃO DA ANIMAÇÃO
        # ============================================
        anim = FuncAnimation(fig, animate, frames=len(t_anim), 
                            interval=30, blit=False, repeat=True)

        # Ajuste do layout principal
        plt.suptitle('Análise Completa da Onda Gaussiana | Animação Simultânea 3D e 2D', 
                    fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout()

        # ============================================
        # INFORMAÇÕES E EXECUÇÃO
        # ============================================
        print("\n" + "="*80)
        print("🎬 ANIMAÇÃO SIMULTÂNEA: GRÁFICO 3D + GRÁFICO 2D")
        print("="*80)
        print(f"Função: u(x,t) = (1/20)·exp(-10·(x-t)²)")
        print(f"\nConfiguração da animação:")
        print(f"  • Ponto animado: segue a curva u({x_const}, t)")
        print(f"  • Intervalo de tempo: t ∈ [{t_min}, {t_max}]")
        print(f"  • Número de frames: {len(t_anim)}")
        print(f"  • Velocidade: 30ms/frame (~33 fps)")
        print(f"\nElementos visuais:")
        print(f"  🔴 Ponto vermelho: Posição atual em ambos os gráficos")
        print(f"  📍 Linha pontilhada: Conexão com o eixo (2D) e plano base (3D)")
        print(f"  🎨 Cor do ponto: Varia com a amplitude (amarelo → vermelho)")
        print(f"\nComportamento:")
        print(f"  • Início (t = 0.00): u = {u(x_const, 0):.6f}")
        print(f"  • Pico   (t = 1.00): u = {u_pico:.6f} ★")
        print(f"  • Fim    (t = 2.00): u = {u(x_const, 2):.6f}")
        print("="*80)
        print("\n▶️ Executando animação...")
        print("   Observe o ponto vermelho se movendo NOS DOIS GRÁFICOS simultaneamente!")
        print("   Feche a janela para encerrar.\n")

        # Mostrar a figura com animação
        plt.show()

        # Opção para salvar a animação (descomente se quiser)
        # from matplotlib.animation import PillowWriter
        # anim.save('animacao_3d_e_2d.gif', writer=PillowWriter(fps=33))
        # print("✓ Animação salva como 'animacao_3d_e_2d.gif'")        

    elif op == 18: #reproducao imagem 2.1 Allen, E. J., Novosel, S. J., & Zhang, Z. (1998)
        dominio = [dominio.Dominio(L=1, L0=0, N=2**(k+2)) for k in range(8)]
        ass = [Assimilacao(dom = dominio[i]) for i in range(8)]
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(4, 2)
        axs[0, 0].plot(dominio[0].x, ass[0].white_noise(), drawstyle='steps-post', label = f"n = {2**2}",linewidth=1)
        axs[0, 0].legend()
        axs[0, 0].set_xlabel("x")

        axs[0, 1].plot(dominio[1].x, ass[1].white_noise(), drawstyle='steps-post', label = f"n = {2**3}", linewidth=1)
        axs[0, 1].legend()
        axs[0, 1].set_xlabel("x")

        axs[1, 0].plot(dominio[2].x, ass[2].white_noise(), drawstyle='steps-post', label = f"n = {2**4}",  linewidth=1)
        axs[1, 0].legend()
        axs[1, 0].set_xlabel("x")

        axs[1, 1].plot(dominio[3].x, ass[3].white_noise(), drawstyle='steps-post', label = f"n = {2**5}",  linewidth=1)
        axs[1, 1].legend()
        axs[1, 1].set_xlabel("x")
        
        axs[2, 0].plot(dominio[4].x, ass[4].white_noise(), drawstyle='steps-post', label = f"n = {2**6}",  linewidth=1)
        axs[2, 0].legend()
        axs[2, 0].set_xlabel("x")

        axs[2, 1].plot(dominio[5].x, ass[5].white_noise(), drawstyle='steps-post', label = f"n = {2**7}",  linewidth=1)
        axs[2, 1].legend()
        axs[2, 1].set_xlabel("x")
        fig.tight_layout()

        axs[3, 0].plot(dominio[6].x, ass[6].white_noise(), drawstyle='steps-post', label = f"n = {2**8}",  linewidth=1)
        axs[3, 0].legend()
        axs[3, 0].set_xlabel("x")

        axs[3, 1].plot(dominio[7].x, ass[7].white_noise(), drawstyle='steps-post', label = f"n = {2**9}",  linewidth=1)
        axs[3, 1].legend()
        axs[3, 1].set_xlabel("x")
        fig.tight_layout()


        
        #fig.legend()
        plt.show() 
    
    elif op == 17:
        ass1 = Assimilacao(dom, modo="estocastico", n_amostras = 2)
        ass2 = Assimilacao(dom, modo="estocastico", n_amostras = 4)
        ass3 = Assimilacao(dom, modo="estocastico", n_amostras = 8)
        ass4 = Assimilacao(dom, modo="estocastico", n_amostras = 16)

        testes = 8
        from rich import print
        from rich.table import Table
        tab = Table(title = "Erro de aproximação")
        tab.add_column("n_iterações", justify = "center")
        tab.add_column("n_amostras = 2", justify = "center")
        tab.add_column("n_amostras = 4", justify = "center")
        tab.add_column("n_amostras = 8", justify = "center")
        tab.add_column("n_amostras = 16", justify = "center")

        for j in range(testes):
            tab.add_row(f"{2**j}",f"{np.mean(ass1.diferenca(iter = 2**j))}", 
                        f"{ass2.diferenca(iter = 2**j)[-1]}", 
                        f"{ass3.diferenca(iter = 2**j)[-1]}", 
                        f"{ass4.diferenca(iter = 2**j)[-1]}")


        print(tab)

    elif op == 16:
        ass1 = Assimilacao(dom, modo="estocastico", n_amostras = 2)
        ass2 = Assimilacao(dom, modo="analitico", ruido= False, n_amostras=2)
        
        ass3 = Assimilacao(dom, modo="estocastico", n_amostras = 4)            
        ass4 = Assimilacao(dom, modo="analitico", ruido= False, n_amostras=4)
        
        ass5 = Assimilacao(dom, modo="estocastico", n_amostras=8)
        ass6 = Assimilacao(dom, modo="analitico", ruido= False, n_amostras = 8)
        
        ass7 = Assimilacao(dom, modo="estocastico", n_amostras=16)
        ass8 = Assimilacao(dom, modo="analitico", ruido= False, n_amostras=16)

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].set_title(f"Métodos com duas amostras e {iteracoes} iterações.")
        axs[0, 0].plot(dom.x, ass1.gradiente_descendente(it = iteracoes), label="Estocástico", linewidth=1)
        axs[0, 0].plot(dom.x, ass2.gradiente_descendente(it = iteracoes), label="Analítico", linewidth=1)
        axs[0, 0].legend()
        axs[0, 0].set_xlabel("x")
        axs[0, 0].set_ylabel("u_0(x)")


        axs[0, 1].set_title(f"Métodos com quatro amostras e {iteracoes} iterações.")
        axs[0, 1].plot(dom.x, ass3.gradiente_descendente(it = iteracoes), label="Estocástico", linewidth=1)
        axs[0, 1].plot(dom.x, ass4.gradiente_descendente(it = iteracoes), label="Analítico", linewidth=1)            
        axs[0, 1].legend()
        axs[0, 1].set_xlabel("x")
        axs[0, 1].set_ylabel("u_0(x)")

        axs[1, 0].set_title(f"Métodos com oito amostras e {iteracoes} iterações.")
        axs[1, 0].plot(dom.x, ass5.gradiente_descendente(it = iteracoes), label="Estocástico", linewidth=1)
        axs[1, 0].plot(dom.x, ass6.gradiente_descendente(it = iteracoes), label="Analítico", linewidth=1)            
        axs[1, 0].legend()
        axs[1, 0].set_xlabel("x")
        axs[1, 0].set_ylabel("u_0(x)")

        axs[1, 1].set_title(f"Métodos com dezesseis amostras e {iteracoes} iterações.")
        axs[1, 1].plot(dom.x, ass7.gradiente_descendente(it = iteracoes), label="Estocástico", linewidth=1)
        axs[1, 1].plot(dom.x, ass8.gradiente_descendente(it = iteracoes), label="Analítico", linewidth=1)            
        axs[1, 1].legend()
        axs[1, 1].set_xlabel("x")
        axs[1, 1].set_ylabel("u_0(x)")
        fig.tight_layout()


        
        #fig.legend()
        plt.show()

    elif op == 15:
        ass1 = Assimilacao(dom, modo="analitico", n_amostras = amos)
        ass2 = Assimilacao(dom, modo="numerico", n_amostras = amos)
        ass3 = Assimilacao(dom, modo="estocastico", n_amostras = amos)
        grad_analitico = ass1.gradiente_descendente()
        grad_numerico = ass2.gradiente_descendente()
        grad_estocastico = ass3.gradiente_descendente()

        plt.figure(figsize=(10, 6))

        plt.plot(dom.x, grad_analitico, label='Gradiente Analítico')
        plt.plot(dom.x, grad_numerico, label='Gradiente Numérico')
        plt.plot(dom.x, grad_estocastico, label='Gradiente Estocástico')

        plt.xlabel('x')
        plt.ylabel('u_0(x)')
        plt.title(f'Assimilação de {amos} amostras e {iteracoes} iterações')
        plt.legend()
        plt.grid(True)

        plt.show()

    elif op == 14: # apresenta os movimentos brownianos em cada amostra
        matriz = ass.matriz_b()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(amos):
            ax.plot(np.full_like(dom.x, i+1), dom.x, matriz[:, i], linewidth=2)

        ax.set_xlabel('Amostra')
        ax.set_ylabel('t')
        ax.set_zlabel('W')
        ax.set_xticks([i for i in range(amos)])
        ax.view_init(25, -60)
        plt.show()

    elif op == 13: #imprimir os passos das amostras
    
        print(ass.passos)

    elif op == 12: #Grafico de todas as amostras
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        
        matriz = ass.matriz_de_amostras()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(amos):
            ax.plot(np.full_like(dom.x, i+1), dom.x, matriz[:, i], linewidth=2)

        ax.set_xlabel('Amostra x_j')
        ax.set_ylabel('t')
        ax.set_zlabel((r'$y_j^{(o)}(t)$'))
        zlabel = ax.zaxis.label
        zlabel.set_rotation(0)   
        ax.set_xticks([i for i in range(amos)])
        ax.view_init(25, -60)
        plt.show()

    elif op == 11: # Constatação do resultado teórico para advecção
            ass1 = Assimilacao(dom, modo="analitico", ruido= True, n_amostras=2)                        
            k = ass1.E
            d1 = ass1.diferenca(iter = iteracoes)
            ass2 = Assimilacao(dom, modo="analitico", ruido= True, n_amostras=4)            
            m = ass2.E
            d2 = ass2.diferenca(iter = iteracoes)
            ass3 = Assimilacao(dom, modo="numérico", ruido= True, n_amostras=2)
            n = ass3.E
            d3 = ass3.diferenca(iter = iteracoes)
            ass4 = Assimilacao(dom, modo="numerico", ruido= True, n_amostras=4)
            d4 = ass4.diferenca(iter = iteracoes)
            ass5 = Assimilacao(dom, modo="analitico", n_amostras= 2)
            d5 = ass5.diferenca(iter = iteracoes)
            ass6 = Assimilacao(dom, modo="analitico",  n_amostras=4) 
            d6 = ass6.diferenca(iter = iteracoes)           
            ass7 = Assimilacao(dom, modo="numérico", n_amostras=2)
            d7 = ass7.diferenca(iter = iteracoes)
            ass8 = Assimilacao(dom, modo="numérico", n_amostras=4)
            d8 = ass8.diferenca(iter = iteracoes)


            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].set_title(f"Método analítico com duas amostras")
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
                             title = "Duas primeiras amostras para assimilação")
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

    elif op == 0: # apresenta o grafico das duas soluções da equação 
        y = sol.solucao_numerica()
        z = sol.solucao_analitica()
        graf = cdg.Grafico2d(x = dom.x,
                             y1 = y,
                             y1_name = "Solução numérica",
                             y2 = z,
                             y2_name = "Solução Analítica",
                             title = "Solução para advecção")
        graf.plot2d()
    
