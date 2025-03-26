import numpy as np
import torch
from snstorch import modules as m
from sns_toolbox.networks import Network
from sns_toolbox.neurons import SpikingNeuron
from sns_toolbox.connections import SpikingSynapse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# SNS Toolbox
dt = 0.01
net = Network()
neuron = SpikingNeuron(membrane_capacitance=5.0, membrane_conductance=1.0,threshold_initial_value=1, reset_potential=-0.1)
neuron = SpikingNeuron(membrane_capacitance=5.0, membrane_conductance=1.0,threshold_initial_value=1, reset_potential=-0.1, threshold_time_constant=5.0, threshold_proportionality_constant=-1.0, threshold_leak_rate=0.5)
neuron = SpikingNeuron(membrane_capacitance=5.0, membrane_conductance=1.0,threshold_initial_value=1, reset_potential=-0.1, threshold_time_constant=5.0, threshold_proportionality_constant=0.0, threshold_leak_rate=0.5, threshold_increment=-0.1)
net.add_neuron(neuron,'Neuron')
net.add_neuron(neuron,'Post')
synapse = SpikingSynapse(max_conductance=1000, reversal_potential=5, time_constant=1, transmission_delay=0, conductance_increment=5)
net.add_connection(synapse, 'Neuron', 'Post')
net.add_input('Neuron')
net.add_output('Neuron', spiking=False)
net.add_output('Neuron', spiking=True)
net.add_output('Post', spiking=False)
net.add_output('Post', spiking=True)
model_snstoolbox = net.compile(dt,backend='torch')

# SNSTorch
tauU = dt/5
params = nn.ParameterDict({
            'tauU': nn.Parameter(torch.tensor(tauU)),
            'leakU': nn.Parameter(torch.tensor(1.0)),
            'restU': nn.Parameter(torch.tensor(0.0)),
            'biasU': nn.Parameter(torch.tensor(0.0)),
            'initU': nn.Parameter(torch.tensor(0.0)),
            'resetU': nn.Parameter(torch.tensor(-0.1)),
            'tauTheta': nn.Parameter(torch.tensor(tauU)),
            'm': nn.Parameter(torch.tensor(0.0)),
            'leakTheta': nn.Parameter(torch.tensor(0.5)),
            'initTheta': nn.Parameter(torch.tensor(1.0)),
            'incTheta': nn.Parameter(torch.tensor(-0.1)),
        })
model_snstorch_pre = m.AdaptiveSpikingLayer(1, params=params)
model_snstorch_post = m.AdaptiveSpikingLayer(1, params=params)
params_syn = nn.ParameterDict({
    'tau': nn.Parameter(torch.tensor([[dt]])),
    'conductanceInc': nn.Parameter(torch.tensor([[5.0]])),
    'reversal': nn.Parameter(torch.tensor([[5.0]]))
})
model_snstorch_syn = m.SpikingChemicalSynapseLinear(1,1, params=params_syn)

num_steps = 1000
t = np.arange(num_steps)*dt
mem_snstoolbox = torch.zeros([num_steps,2])
spk_snstoolbox = torch.zeros([num_steps,2])
mem_snstorch = torch.zeros([num_steps,2])
spk_snstorch = torch.zeros([num_steps,2])
theta_snstoolbox = torch.zeros([num_steps,2])
theta_snstorch = torch.zeros([num_steps,2])
syn_snstoolbox = torch.zeros(num_steps)
syn_snstorch = torch.zeros(num_steps)
inp_pre = torch.tensor([2.0])
inp_post = torch.tensor([0.0])
state_pre = torch.zeros(1)
theta_pre = torch.ones(1)
syn = torch.zeros(1)
syn_old = torch.zeros(1)
state_syn = torch.zeros(1)
state_post = torch.zeros(1)
theta_post = torch.ones(1)
spikes_pre = torch.zeros(1)
for i in range(num_steps):
    out = model_snstoolbox(inp_pre)
    mem_snstoolbox[i,0] = out[0]
    spk_snstoolbox[i,0] = out[1]
    mem_snstoolbox[i, 1] = out[2]
    spk_snstoolbox[i, 1] = out[3]
    theta_snstoolbox[i,:] = model_snstoolbox.theta
    syn_snstoolbox[i] = model_snstoolbox.g_spike[1,0]
    # if out[1] > 0:
    #     print('Spike')

    spikes_pre, state_pre, theta_pre = model_snstorch_pre(inp_pre, state_pre, theta_pre)
    syn, state_syn = model_snstorch_syn(spikes_pre, state_syn, state_post)
    spikes_post, state_post, theta_post = model_snstorch_post(syn, state_post, theta_post)


    syn_old = syn

    mem_snstorch[i,0] = state_pre
    mem_snstorch[i,1] = state_post
    spk_snstorch[i,0] = spikes_pre
    spk_snstorch[i,1] = spikes_post
    theta_snstorch[i,0] = theta_pre
    theta_snstorch[i,1] = theta_post
    syn_snstorch[i] = state_syn

plt.figure()
plt.subplot(7,1,1)
plt.title('Pre Membrane')
plt.plot(t, mem_snstoolbox[:,0], label='SNS-Toolbox')
plt.plot(t, mem_snstorch.detach()[:,0], label='SNSTorch')
plt.legend()
plt.subplot(7,1,2)
plt.title('Post Membrane')
plt.plot(t, mem_snstoolbox[:,1], label='SNS-Toolbox')
plt.plot(t, mem_snstorch.detach()[:,1], label='SNSTorch')
plt.legend()
plt.subplot(7,1,3)
plt.title('Pre Spikes')
plt.plot(t, spk_snstoolbox[:,0], label='SNS-Toolbox')
plt.plot(t, spk_snstorch.detach()[:,0], label='SNSTorch')
plt.legend()
plt.subplot(7,1,4)
plt.title('Post Spikes')
plt.plot(t, spk_snstoolbox[:,1], label='SNS-Toolbox')
plt.plot(t, spk_snstorch.detach()[:,1], label='SNSTorch')
plt.legend()
plt.subplot(7,1,5)
plt.title('Pre Theta')
plt.plot(t, theta_snstoolbox[:,0], label='SNS-Toolbox')
plt.plot(t, theta_snstorch.detach()[:,0], label='SNSTorch')
plt.legend()
plt.subplot(7,1,6)
plt.title('Post Theta')
plt.plot(t, theta_snstoolbox[:,1], label='SNS-Toolbox')
plt.plot(t, theta_snstorch.detach()[:,1], label='SNSTorch')
plt.legend()
plt.subplot(7,1,7)
plt.title('Synapse')
plt.plot(t, syn_snstoolbox, label='SNS-Toolbox')
plt.plot(t, syn_snstorch.detach(), label='SNSTorch')
plt.legend()
plt.xlabel('t (ms)')

plt.show()
