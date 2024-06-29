import numpy as np
from neurodiffeq import diff
from neurodiffeq.ode import solve
from neurodiffeq.conditions import IVP
from neurodiffeq.generators import Generator1D, ConcatGenerator
from neurodiffeq.networks import FCNN
from torch import nn
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.optim as optim
import time


columns = ['DE_name', 'ActFunc_name', 'AvgNormInf', 'AvgNorm1','Avgloss','Elps_time']
FinalDF = pd.DataFrame(columns=columns)

Dtest = 10000;
ts = np.linspace(0, 1, Dtest)

# Define Equations

#Equation with alpha=1
DE42 = lambda u, t:  diff(u, t, order=2) + (1 / t) * diff(u, t, order=1)+torch.exp(u)
Init_Val42 = IVP(t_0=0.0, u_0=0.0, u_0_prime=0.0)
Ana42 = (-1/4)*(ts**2)+(1/64)*(ts**4)-(1/768)*(ts**6)+(1/8192)*(ts**8)

#Equation with alpha=2
DE43 = lambda u, t:  diff(u, t, order=2) + (2 / t) * diff(u, t, order=1)+torch.exp(u)
Init_Val43 = IVP(t_0=0.0, u_0=0.0, u_0_prime=0.0)
Ana43 = (-1/6)*(ts**2)+(1/120)*(ts**4)-(1/1890)*(ts**6)+(61/1632960)*(ts**8)-(629/224532000)*(ts**10)

#Equation with alpha=3
DE44 = lambda u, t:  diff(u, t, order=2) + (3 / t) * diff(u, t, order=1)+torch.exp(u)
Init_Val44 = IVP(t_0=0.0, u_0=0.0, u_0_prime=0.0)
Ana44 = (-1/8)*(ts**2)+(1/192)*(ts**4)-(5/18432)*(ts**6)+(23/1474560)*(ts**8)

#Define Dictionaries

DEsDic ={'DE42': DE42, 'DE43': DE43, 'DE44': DE44}

InitValsDic = { 'DE42': Init_Val42, 'DE43': Init_Val43, 'DE44': Init_Val44 }

AnaSolDic = {'DE42': Ana42, 'DE43': Ana43, 'DE44': Ana44}

activeFunctionsDic = {'GELU': nn.GELU, "ReLU":nn.ReLU, "SiLU": nn.SiLU,
                      "PReLU": nn.PReLU,"CELU": nn.CELU, "ELU": nn.ELU,"Tanhshrink": nn.Tanhshrink,
                       "SELU": nn.SELU}

for deIndex, defun in DEsDic.items():
    u_ana = AnaSolDic[deIndex]
    for actind, actfun in activeFunctionsDic.items():
        newrow = []
        
        net = FCNN(n_input_units=1, n_output_units=1, hidden_units=[32, 32, 32], actv=actfun)

        optimizer = optim.Adam(net.parameters())
        try:
            newrow.extend([deIndex, actind])
            sum_inf = 0
            sum_l1 = 0
            sum_loss = 0
            start_time = time.perf_counter()
            for it in range(30):
                train_gen = Generator1D(size=32, t_min=0.0, t_max=1.0, method='uniform')
                solution_ex, loss_ex = solve(ode=defun, condition=InitValsDic[deIndex], t_min=0.0, t_max=1,
                             net=net, max_epochs=3000, train_generator=train_gen, optimizer=optimizer)

                u_net = solution_ex(ts, to_numpy=True)

                sum_inf += np.amax(np.abs((u_ana - u_net)))

                diff_ana_net = abs(u_ana - u_net)
                sum_l1 += sum(diff_ana_net) / Dtest

                sum_loss += (loss_ex['train_loss'][2999])

            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) / 30
            norm_inf = sum_inf / 30
            norm_l1 = sum_l1 / 30
            loss = sum_loss / 30

            newrow.extend([norm_inf, norm_l1, loss, elapsed_time])
            del(net)
            del(optimizer)

            FinalDF.loc[len(FinalDF)] = newrow
        except Exception as e:
            print("there is and error in", str(e))
    FinalDF.to_excel("E://FinalResults.xlsx", index=False)


