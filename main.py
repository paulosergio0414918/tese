import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import aguas_rasas_linear as swel
import aguas_rasas_nao_linear as swenl
import dominio

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
