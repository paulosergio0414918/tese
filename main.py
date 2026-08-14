import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hashlib
import json
from pathlib import Path
import aguas_rasas_linear as swel
import aguas_rasas_nao_linear as swenl
import dominio

op = 14
iteracoes = 2**3

#### Variáveis
# N=1025; M = 513 #cfl = 0.5
# N=1024; M = 320 #cfl = 0.8
N=512;  M = 160 #cfl = 0.8
# N=1025; M=257   #cfl = 1
amos = 2
ruido = False
first_sample = 1 # paper uses first_sample = 0.2
Delta_x =  0.09 # paper uses Delta_x = 0.09 end Delta_x = 0.375 for counter-example
#discretizacao = "godunov_euler"
#discretizacao = "muscl_ssprk33"
discretizacao = "malha_c"
modo = "malha_c"
#modo = "analitico"
save = True



if op == 14:

    params_dict = {
        "model" : 'swe_l_op_14',
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

    folder = Path("./data") # identify the folder

    save_path = folder / f"{hash_code}.npz" 

    with np.load(save_path) as dados:
        info = dados['info'],
        erro2 = dados['erro2'],
        erro3 = dados['erro3'],
        erro4 = dados['erro4'],
        erro5 = dados['erro5'],
        erro6 = dados['erro6']

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
