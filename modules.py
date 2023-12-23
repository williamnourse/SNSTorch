import torch
import torch.nn as nn

class NonSpikingLayer(nn.Module):
    def __init__(self, size,
                 tau=None, leak=None, rest=None, bias=None, init=None, generator=None, device=None, dtype=torch.float32,
                 ):
        super().__init__()
        if device is None:
            device = 'cpu'
        if tau is None:
            tau = torch.rand(size, dtype=dtype, generator=generator)
        if leak is None:
            leak = torch.rand(size, dtype=dtype, generator=generator)
        if rest is None:
            rest = torch.zeros(size, dtype=dtype)
        if bias is None:
            bias = torch.rand(size, dtype=dtype, generator=generator)
        if init is None:
            init = rest.clone().detach()

        self.tau = nn.Parameter(tau.to(device))
        self.leak = nn.Parameter(leak.to(device))
        self.rest = nn.Parameter(rest.to(device))
        self.bias = nn.Parameter(bias.to(device))
        self.state_0 = nn.Parameter(init.to(device))

    def forward(self,x=None,state_prev=None):
        if state_prev is None:
            state_prev = self.state_0
        if x is None:
            state = state_prev + self.tau * (-self.leak * (state_prev - self.rest) + self.bias)
        else:
            state = state_prev + self.tau * (-self.leak * (state_prev - self.rest) + self.bias + x)
        return state

class PiecewiseActivation(nn.Module):
    def __init__(self, min_val=0, max_val=1):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self,x):
        return torch.clamp((x-self.min_val)/(self.max_val-self.min_val),0,1)

class ChemicalSynapse(nn.Module):
    def __init__(self, size_pre, size_post, max_conductance=None, reversal=None, activation=PiecewiseActivation, device=None):
        super().__init__()
        if device is None:
            device = 'cpu'
        if max_conductance is None:
            max_conductance = torch.rand([size_post,size_pre])
        if reversal is None:
            reversal = 2*torch.rand([size_post,size_pre])-1
        self.max_conductance = nn.Parameter(max_conductance.to(device))
        self.reversal = nn.Parameter(reversal.to(device))
        self.activation = activation()

    def forward(self, state_pre, state_post):
        activated_pre = self.activation(state_pre)
        # batch_max_conductance =
        if state_pre.dim()>1:
            conductance = self.max_conductance * activated_pre.unsqueeze(1)
            left = torch.sum(conductance * self.reversal, dim=2)
            right = state_post * torch.sum(conductance, dim=2)
        else:
            conductance = self.max_conductance * activated_pre
            left = torch.sum(conductance * self.reversal,dim=1)
            right = state_post*torch.sum(conductance,dim=1)
        out = left-right
        return out

# Example usage
# rows, cols = 2, 2
# model = NonSpikingLayer(rows * cols)
# synapse = ChemicalSynapse(4, 2)
#
# # Example single input
# input_data_single = torch.randn([rows * cols])
# output_single = model(input_data_single)
# synapse_output_single = synapse(output_single, model.state_0[:2])
# print(synapse_output_single)
#
# # Example batch input
# input_data_batch = torch.randn([3, rows * cols])
# output_batch = model(input_data_batch)
# state_batch = model.state_0[:2].unsqueeze(0).repeat(3,1)
# synapse_output_batch = synapse(output_batch, state_batch)
# print(synapse_output_batch)
