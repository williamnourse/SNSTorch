import numpy as np
from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingMatrixConnection
from modules import NonSpikingLayer, ChemicalSynapse
import matplotlib.pyplot as plt
import torch
import time

"""General Parameters"""
num_neurons_pre = 1000
num_neurons_post = 200
dt = 0.01
t_max = 10

"""SNS-Toolbox"""
net = Network()
neuron_model = NonSpikingNeuron(membrane_conductance=1,membrane_capacitance=5,resting_potential=0)
net.add_population(neuron_model,shape=[num_neurons_pre],name='Source',initial_value=torch.zeros(num_neurons_pre))
net.add_input('Source',num_neurons_pre)
net.add_output('Source')

net.add_population(neuron_model,shape=[num_neurons_post],name='Dest',initial_value=torch.zeros(num_neurons_post))
net.add_output('Dest')
g = torch.rand([num_neurons_post,num_neurons_pre])
rev = 2*torch.rand([num_neurons_post,num_neurons_pre])-1
synapse = NonSpikingMatrixConnection(max_conductance=g.numpy(), reversal_potential=rev.numpy(),
                                     e_lo=np.zeros([num_neurons_post,num_neurons_pre]),
                                     e_hi=np.ones([num_neurons_post,num_neurons_pre]))
net.add_connection(synapse,'Source','Dest')

model_toolbox = net.compile(dt=dt,backend='torch',device='cuda')

"""SNSTorch"""
class ModelTorch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        tau_pre = dt / 5.0 * torch.ones(num_neurons_pre)
        leak_pre = torch.ones(num_neurons_pre)
        rest_pre = torch.zeros(num_neurons_pre)
        bias_pre = torch.zeros(num_neurons_pre)
        init_pre = torch.zeros(num_neurons_pre)

        tau_post = dt / 5.0 * torch.ones(num_neurons_post)
        leak_post = torch.ones(num_neurons_post)
        rest_post = torch.zeros(num_neurons_post)
        bias_post = torch.zeros(num_neurons_post)
        init_post = torch.zeros(num_neurons_post)

        self.pre = NonSpikingLayer(num_neurons_pre, tau_pre, leak_pre, rest_pre, bias_pre, init_pre)
        self.post = NonSpikingLayer(num_neurons_post, tau_post, leak_post, rest_post, bias_post, init_post)
        self.synapse = ChemicalSynapse(num_neurons_pre,num_neurons_post, max_conductance=g, reversal=rev)

    def forward(self, x, state_pre, state_post):
        state_synapse = self.synapse(state_pre,state_post)
        state_pre = self.pre(x,state_pre)
        state_post = self.post(state_synapse,state_post)
        return state_pre,state_post


"""Simulation"""
model_torch = ModelTorch().to('cuda')
x = torch.rand(num_neurons_pre)

t = np.arange(0, t_max, dt)
data_toolbox = np.zeros([len(t), num_neurons_pre+num_neurons_post])
data_torch = torch.zeros([len(t), num_neurons_pre+num_neurons_post]).to('cuda')
state_pre = model_torch.pre.state_0
state_post = model_torch.post.state_0
# Run for all steps
start = time.time()
# with torch.no_grad():
for i in range(len(t)):
    data_toolbox[i, :] = model_toolbox(x)
    # state = model_torch(x,state)
    # data_torch[i,:] = state
end = time.time()
print('SNS-Toolbox: %f'%(end-start))

x = x.to('cuda')
start = time.time()
with torch.no_grad():
    for i in range(len(t)):
        # data_toolbox[i, :] = model_toolbox(x)
        state_pre, state_post = model_torch(x,state_pre,state_post)
        data_torch[i,:num_neurons_pre] = state_pre
        data_torch[i,num_neurons_pre:] = state_post
end = time.time()
print('SNSTorch: %f'%(end-start))
data_torch = data_torch.to('cpu')

data_toolbox = data_toolbox.transpose()
data_torch = data_torch.numpy().transpose()

"""Comparison"""
plt.figure()
plt.subplot(2,2,1)
plt.title('SNS-Toolbox (Pre)')
for i in range(num_neurons_pre):
    plt.plot(t, data_toolbox[i, :])
plt.subplot(2,2,2)
plt.title('SNS-Toolbox (Post)')
for i in range(num_neurons_post):
    plt.plot(t, data_toolbox[i+num_neurons_pre, :])
plt.subplot(2,2,3)
plt.title('SNSTorch (Pre)')
for i in range(num_neurons_pre):
    plt.plot(t, data_torch[i, :])
plt.subplot(2,2,4)
plt.title('SNSTorch (Post)')
for i in range(num_neurons_post):
    plt.plot(t, data_torch[i+num_neurons_pre, :])

plt.figure()
plt.title('Difference')
for i in range(num_neurons_pre+num_neurons_post):
    plt.plot(t, data_toolbox[i, :]-data_torch[i, :])
plt.ylim([-1,1])

plt.show()
