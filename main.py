import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import aguas_rasas_linear as swel
import aguas_rasas_nao_linear as swenl
import dominio

dom = dominio.Dominio(N=1024, M=256) #cfl = 1
#dom = dominio.Dominio(N=1024, M=320) #cfl = 0.8
sol_nl = swenl.SolucaoAguasRasasNaoLinear(dom, cfl= 1)
sol_l = swel.SolucaoAguasRasas(dom)

for i in tqdm(range(256)):

    y = sol_l.solucao_numerica(modo = "muscl_ssprk33", tempo = i)['eta']
    z = sol_nl.solucao_numerica(modo = "muscl_ssprk33", tempo = i)['eta']
    e_l = np.linalg.norm(y)
    e_nl = np.linalg.norm(z)             

    plt.clf()
    plt.xlim(0, 2.3)
    plt.ylim(0, 0.03) 
    plt.grid()
    plt.plot(dom.x, y, label = 'Linear SWE ' )
    plt.plot(dom.x, z, label = 'Nonlinear SWE ', linestyle='--')
    #plt.title(f'Execução {i+1} de {256} do modelo {discretizacao} com cfl = {cfl}.')
    plt.legend()

    #plt.show(block = False)
    plt.pause(0.001)

plt.show()
