import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hashlib
import json
from pathlib import Path
import aguas_rasas_linear as swel
import aguas_rasas_nao_linear as swenl
from dominio import Dominio


#opc = 'custo'
opc = "all_solutions"
#opc = "erro"

otim = 'ot' # caso queira gradimente descendente otimizado
#otim = no # caso queira gradiente descendente não otimizado

###opção
op = 18
iteracoes = 2**6

#### Variáveis
# N=1025; M = 513 #cfl = 0.5
N=1024; M = 320 #cfl = 0.8
#N=512;  M = 160 #cfl = 0.8
#N=1025; M=257   #cfl = 1
amos = 2
ruido = False
first_sample = .2 # paper uses first_sample = 0.2
Delta_x =  0.09 # paper uses Delta_x = 0.09 end Delta_x = 0.375 for counter-example

discretizacao = "malha_c"
modo = "malha_c"
#modo = "analitico"

eq = 'nao_linear'

dom = Dominio(N = N, M = M) #cfl = 0.8
sol = swel.SolucaoAguasRasas(dom)

flag2 = 'with_noise' if ruido else 'without_noise'

def _tag(v): return f"{v:g}".replace("-", "m").replace(".", "p")

def _load(eq, n_obs,otim , opc, modo, fs, Dx, it):
    return np.loadtxt(f"data/{eq}_gd_n{n_obs}_{otim}_{opc}_{modo}_fs_{_tag(fs)}_deltax_{_tag(Dx)}_it_{it}_{flag2}.csv", delimiter=",")

def _load_antigo(n_obs,otim , opc, modo, fs, Dx, it):
    return np.loadtxt(f"data/dados_antigos/gd_n{n_obs}_{otim}_{opc}_{modo}_fs_{_tag(fs)}_deltax_{_tag(Dx)}_it_{it}.csv", delimiter=",")

cores = {2: "black", 3: "blue", 4: "green", 5: "red", 6: "yellow"}

if opc == "custo":
    fig, ax = plt.subplots()
    x = np.arange(1, iteracoes + 1)
    
    for n in (2, 3, 4, 5, 6):
        c_ot = _load(eq, n, 'ot', opc, modo, first_sample, Delta_x, iteracoes )
        #c_ot_ant = _load_antigo(n, 'ot', opc, modo, first_sample, Delta_x, iteracoes )
        #c_no = _load(n, 'no', opc, modo, first_sample, Delta_x, iteracoes )
        ax.scatter(x, c_ot/c_ot[0], s=8, color=cores[n], label=f"otimizado {n} amostras")
        #ax.scatter(x, c_ot_ant/c_ot_ant[0], s=8, label=f"otimizado {n} amostras antigo")
        #ax.scatter(x, c_no/c_no[0], s=8, color=cores[n], label=f"não otimizado {n} amostras")

    ax.set_yscale("log")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("J^(n) / J^(0)")
    ax.set_title(f"Convergência do custo {modo} — Δx = {Delta_x}")
    ax.legend()
    plt.tight_layout()
    plt.show()

elif opc == "erro":
    fig, ax = plt.subplots()
    x = np.arange(1, iteracoes + 1)

    for n in (2,3,4,5, 6):
        c_ot = _load(eq, n, 'ot', opc, modo, first_sample, Delta_x, iteracoes )
        #c_ot_ant = _load_antigo(n, 'ot', opc, modo, first_sample, Delta_x, iteracoes )
        #c_no = _load(n, 'no', opc, modo, first_sample, Delta_x, iteracoes )
        ax.scatter(x, c_ot, s=8, color=cores[n], label=f"otimizado {n} amostras")
        #ax.scatter(x, c_ot_ant/c_ot_ant[0], s=8, label=f"otimizado {n} amostras antigo")
        #ax.scatter(x, c_no, s=8, color=cores[n], label=f"não otimizado {n} amostras")

    ax.set_yscale("log")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("||(phi^t(x) - phi^n(x))||/||phi^t(x)||")
    ax.set_title(f"Erro {modo} de reconstrução da condição inicial— Δx = {Delta_x}")
    ax.legend()
    plt.tight_layout()
    plt.show()

elif opc == "all_solutions":
    if eq == 'linear':
        all_sol2 = _load(eq, amos, "ot", opc, "analitico", first_sample, Delta_x, iteracoes)
        
    all_sol = _load(eq, amos, "ot", opc, modo, first_sample, Delta_x, iteracoes)
    m = all_sol.shape[1]
    x= np.linspace(-4,4,m)
    for j in range(iteracoes):
        eta_j = all_sol[j, :]
        #eta2_j = all_sol2[j, :]

        plt.clf()
        plt.ylim(-0.025, 0.06)
        plt.xlim(-2.3, 2.3)
        plt.plot(x, sol.eta_zero(), label='φ^(t)')
        #plt.plot(x, eta2_j, label=f"φ^(n) analitico")
        plt.plot(x, eta_j, label=f"φ^(n) {modo}")


        plt.title(
            f"Iteração {j+1}/{iteracoes } — "
            f"n={amos}, Δx={Delta_x}, fs={first_sample}, modo={modo}"
        )
        plt.legend()
        plt.pause(0.1)

    plt.show()






'''


op = 14
iteracoes = 8

#### Variáveis
# N=1025; M = 513 #cfl = 0.5
N=1024; M = 320 #cfl = 0.8
#N=512;  M = 160 #cfl = 0.8
# N=1025; M=257   #cfl = 1
amos = 2
ruido = False
first_sample = .2 # paper uses first_sample = 0.2
Delta_x =  0.09 # paper uses Delta_x = 0.09 end Delta_x = 0.375 for counter-example
#discretizacao = "godunov_euler"
#discretizacao = "muscl_ssprk33"
discretizacao = "malha_c"
modo = "malha_c"
#modo = "analitico"
save = True



if op == 14:

    params_dict = {
        "model" : 'swe_l_all',
        "M" : M,
        "N" : N,
        "it" : iteracoes,
        "noise" : ruido,
        "first_sample" : first_sample,
        "Delta_x" :  Delta_x,
        "discretization": discretizacao,
        "grad": modo 
        }

    params_string = json.dumps(params_dict, sort_keys=True) # create an javascript string

    hash_code = hashlib.md5(params_string.encode('utf-8')).hexdigest() # criate a name to the file
    print(hash_code)
    folder = Path("./data") # identify the folder

    save_path = folder / f"{hash_code}.npz" 

    with np.load(save_path) as dados:
        info = dados['info'],
        gd2_erro = dados['error'],
        gd2_custo = dados['custo'],
        gd2_alpha = dados['alpha'],
        gd2_all_solutions = dados['all_solutions'],

    print(info[0])

elif op == -1: #gráfico dos dois modelos 



    #dom = dominio.Dominio(N=1024, M=256) #cfl = 1
    dom = dominio.Dominio(N=1024, M=320) #cfl = 0.8
    sol_nl = swenl.SolucaoAguasRasasNaoLinear(dom)
    sol_l = swel.SolucaoAguasRasas(dom)
    tempo_linear = 0
    tempo_nao_linear = 0
    m = dom.M
    flag = 0

    for i in tqdm(range(m+50)):
        if i < m:
            y = sol_l.solucao_numerica(modo = "malha_c", tempo = i)['eta']
            tempo_linear += dom.dt
        graf = sol_nl.solucao_numerica(modo = "malha_c", tempo = i)
        z = graf['eta']
        tempo_nao_linear += graf['time']
        plt.clf()
        plt.xlim(0, 2.3)
        plt.ylim(0, 0.03) 
        plt.grid()
        plt.plot(dom.x, y, label = f'Linear SWE t = {tempo_linear:.4f}' )
        plt.plot(dom.x, z, label = f'Nonlinear SWE  t = {tempo_nao_linear:.4f}')
        #plt.title(f'Execução {i+1} de {256} do modelo {discretizacao} com cfl = {cfl}.')
        plt.legend()

        #plt.show(block = False)
        plt.pause(0.001)
        if tempo_nao_linear > tempo_linear:
            break


    plt.show()
'''