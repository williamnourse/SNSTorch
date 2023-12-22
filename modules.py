import torch
import torch.nn as nn

class NonSpikingLayer(nn.Module):
    def __init__(self, size,
                 tau=None, leak=None, rest=None, bias=None, init=None, generator=None, device=None, dtype=torch.float32):
        super().__init__()
        if device is None:
            device = 'cpu'
        if tau is None:
            tau = torch.rand(size, dtype=dtype, generator=generator, device=device)
        if leak is None:
            leak = torch.rand(size, dtype=dtype, generator=generator, device=device)
        if rest is None:
            rest = torch.zeros(size, dtype=dtype, device=device)
        if bias is None:
            bias = torch.rand(size, dtype=dtype, generator=generator, device=device)
        if init is None:
            init = rest.clone().detach()

        self.tau = nn.Parameter(tau)
        self.leak = nn.Parameter(leak)
        self.rest = rest
        self.bias = nn.Parameter(bias)
        self.state_0 = init

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
    def __init__(self, size_pre, size_post, max_conductance=None, reversal=None, act=PiecewiseActivation):
        super().__init__()
        if max_conductance is None:
            max_conductance = torch.rand([size_post,size_pre])
        if reversal is None:
            reversal = 2*torch.rand([size_post,size_pre])-1
        self.max_conductance = max_conductance
        self.reversal = reversal
        self.act = act()

    def forward(self, state_pre, state_post):
        conductance = self.max_conductance*self.act(state_pre)
        left = torch.einsum('ij,ij->i', conductance,self.reversal)
        right = torch.einsum('ij,i->i',conductance,state_post)
        return left-right
