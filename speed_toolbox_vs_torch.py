from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingPatternConnection
import snstorch.modules as m
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

def gen_sns(shape, shape_flat):
    R = 20.0  # range of network activity (mV)
    neuron_type = NonSpikingNeuron()  # generic neuron type
    net = Network(name='Visual Network')  # create an empty network

    # Retina
    net.add_population(neuron_type, shape, name='Retina')  # add a 2d population the same size as the scaled image
    net.add_input('Retina', size=shape_flat, name='Image')  # add a vector input for the flattened scaled image
    net.add_output('Retina', name='Retina Output')  # add a vector output from the retina, scaled correctly

    # Lamina
    net.add_population(neuron_type, shape, name='Lamina')

    del_e_ex = 160.0  # excitatory reversal potential
    del_e_in = -80.0  # inhibitory reversal potential
    k_ex = 1.0  # excitatory gain
    k_in = -1.0 / 9.0  # inhibitory gain
    g_max_ex = (k_ex * R) / (del_e_ex - k_ex * R)  # calculate excitatory conductance
    g_max_in = (k_in * R) / (del_e_in - k_in * R)  # calculate inhibitory conductance

    g_max_kernel = np.array([[g_max_in, g_max_in, g_max_in],  # kernel matrix of synaptic conductances
                             [g_max_in, g_max_ex, g_max_in],
                             [g_max_in, g_max_in, g_max_in]])
    del_e_kernel = np.array([[del_e_in, del_e_in, del_e_in],  # kernel matrix of synaptic reversal potentials
                             [del_e_in, del_e_ex, del_e_in],
                             [del_e_in, del_e_in, del_e_in]])
    e_lo_kernel = np.zeros([3, 3])
    e_hi_kernel = np.zeros([3, 3]) + R
    connection_hpf = NonSpikingPatternConnection(g_max_kernel, del_e_kernel, e_lo_kernel,
                                                 e_hi_kernel)  # pattern connection (acts as high pass filter)
    net.add_connection(connection_hpf, 'Retina', 'Lamina', name='HPF')  # connect the retina to the lamina
    net.add_output('Lamina', name='Lamina Output')  # add a vector output from the lamina
    return net

class TorchModel(nn.Module):
    def __init__(self, shape, shape_flat, device):
        super().__init__()
        self.shape = shape
        self.shape_flat = shape_flat

        self.retina = m.NonSpikingLayer(shape, device=device)
        self.syn = m.NonSpikingChemicalSynapseConv(1,1,3, device=device)
        self.lamina = m.NonSpikingLayer([shape[0]-2, shape[1]-2], device=device)

        self.syn.setup()

    def init(self):
        state_retina = self.retina.params['init']
        state_lamina = self.lamina.params['init']
        return state_retina, state_lamina

    def forward(self, x, state_retina, state_lamina):
        syn = self.syn(state_retina, state_lamina)

        state_retina = self.retina(x, state_retina)
        state_lamina = self.lamina(syn, state_lamina)

        return state_retina, state_lamina

start = time.time()
size = [4,8,16,32,64,128]
times_sns_cpu = torch.zeros(len(size))
times_sns_gpu = torch.zeros(len(size))
times_torch_cpu = torch.zeros(len(size))
times_torch_gpu = torch.zeros(len(size))
num_steps = 5
with torch.no_grad():
    for i in range(len(size)):
        print(size[i])
        shape = [size[i], size[i]]
        shape_flat = size[i]*size[i]
        stim = torch.rand(shape)

        # sns_toolbox
        net_sns = gen_sns(shape, shape_flat)
        if size[i] <= 64:
            print('SNS CPU')
            model = net_sns.compile(backend='torch', device='cpu')
            timer = 0
            for j in range(2*num_steps):
                if j == num_steps:
                    timer = time.time()
                _ = model(stim.flatten())
            val = time.time()-timer
            times_sns_cpu[i] = val
            print('%.4f sec'%val)
        stim = stim.to('cuda')
        if size[i] <= 64:
            print('SNS GPU')
            model = net_sns.compile(backend='torch', device='cuda')
            timer = 0
            for j in range(2 * num_steps):
                if j == num_steps:
                    timer = time.time()
                _ = model(stim.flatten())
            val = time.time() - timer
            times_sns_gpu[i] = val
            print('%.4f sec'%val)

        if size[i] <= 128:
            #snstorch
            print('Torch GPU')
            model = TorchModel(shape, shape_flat, 'cuda')
            state_retina, state_lamina = model.init()
            timer = 0
            for j in range(2 * num_steps):
                if j == num_steps:
                    timer = time.time()
                state_retina, state_lamina = model(stim, state_retina, state_lamina)
            val = time.time()-timer
            times_torch_gpu[i] = val
            print('%.4f sec'%val)

        print('Torch CPU')
        model = TorchModel(shape, shape_flat, 'cpu')
        stim = stim.to('cpu')
        state_retina, state_lamina = model.init()
        timer = 0
        for j in range(2 * num_steps):
            if j == num_steps:
                timer = time.time()
            state_retina, state_lamina = model(stim, state_retina, state_lamina)
        val = time.time()-timer
        times_torch_cpu[i] = val
        print('%.4f sec'%val)

data = {'snsCPU': times_sns_cpu,
        'snsGPU': times_sns_gpu,
        'torchCPU': times_torch_cpu,
        'torchGPU': times_torch_gpu}
pickle.dump(data, open('speed_comparison.p', 'wb'))
