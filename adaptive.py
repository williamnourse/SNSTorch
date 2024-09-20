from snstorch import modules as m
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
c = 0.1
k = (1-c)/c
tauA = 50
tauU = 5
print(4/(tauA*tauA+tauU*tauU+2*tauU*tauA+3))
params = nn.ParameterDict({
            'tauU': nn.Parameter(torch.tensor(0.1/tauU)),
            'leakU': nn.Parameter(torch.tensor(1.0)),
            'restU': nn.Parameter(torch.tensor(0.0)),
            'biasU': nn.Parameter(torch.tensor(0.0)),
            'initU': nn.Parameter(torch.tensor(0.0)),
            'tauA': nn.Parameter(torch.tensor(0.1/tauA)),
            'leakA': nn.Parameter(torch.tensor(1.0)),
            'restA': nn.Parameter(torch.tensor(0.0)),
            'gainA': nn.Parameter(torch.tensor(k)),
            'initA': nn.Parameter(torch.tensor(1.0))
        })

model = m.AdaptiveNonSpikingLayer(1, params=params)

state = torch.zeros(2000)
adapt = torch.zeros_like(state)
x = torch.tensor(1.0)

for i in range(1,len(state)):
    state[i], adapt[i] = model(x, state[i-1], adapt[i-1])

plt.figure()
plt.axhline(y=c,color='black')
plt.plot(state.detach().numpy())
plt.plot(adapt.detach().numpy())
plt.show()