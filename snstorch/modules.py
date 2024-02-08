import torch
import torch.nn as nn
import torch.autograd.profiler as profiler
import torch.jit as jit
from typing import List

class NonSpikingLayer(jit.ScriptModule):
    def __init__(self, size, params=None, generator=None, device=None, dtype=torch.float32,
                 ):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.params = nn.ParameterDict({
            'tau': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'leak': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'rest': nn.Parameter(torch.zeros(size, dtype=dtype).to(device)),
            'bias': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'init': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device))
        })
        if params is not None:
            self.params.update(params)

    @jit.script_method
    def forward(self, x, state):
        # if state is None:
        #     state = self.params['init']
        # with profiler.record_function("NEURAL UPDATE"):
        state = state + self.params['tau'] * (-self.params['leak'] * (state - self.params['rest']) + self.params['bias'] + x)
            # state += new_state
        # if x is not None:
        #     state += self.params['tau']*x
        return state

class ClampActivation(jit.ScriptModule):
    def __init__(self):
        super().__init__()

    @jit.script_method
    def forward(self, x):
        return torch.clamp(x,0,1)

class PiecewiseActivation(jit.ScriptModule):
    def __init__(self, min_val=0, max_val=1):
        super().__init__()
        self.min_val = min_val
        self.inv_range = 1/(max_val-min_val)

    @jit.script_method
    def forward(self,x):
        # with profiler.record_function("PIECEWISE SIGMOID"):
            # x -= self.min_val
            # x *= self.inv_range
            # x.clamp_(0,1)
        return torch.clamp((x - self.min_val) * self.inv_range, 0, 1)
        # return x
        # return torch.clamp((x-self.min_val)/self.range,0,1)

class NonSpikingChemicalSynapseLinear(jit.ScriptModule):
    def __init__(self, size_pre, size_post, params=None, activation=ClampActivation, device=None,
                 dtype=torch.float32, generator=None):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.rand([size_post,size_pre],dtype=dtype, generator=generator).to(device)),
            'reversal': nn.Parameter((2*torch.rand([size_post,size_pre], generator=generator)-1).to(device))
        })
        if params is not None:
            self.params.update(params)
        self.activation = activation()

    @jit.script_method
    def forward(self, state_pre, state_post):
        # with profiler.record_function("LINEAR SYNAPSE"):
        activated_pre = self.activation(state_pre)
        if state_pre.dim() > 1:
            conductance = self.params['conductance'] * activated_pre.unsqueeze(1)
        else:
            conductance = self.params['conductance'] * activated_pre
        if conductance.dim()>2:
            left = torch.sum(conductance * self.params['reversal'], dim=2)
            right = state_post * torch.sum(conductance, dim=2)
        else:
            left = torch.sum(conductance * self.params['reversal'],dim=1)
            right = state_post*torch.sum(conductance,dim=1)
        out = left-right
        return out

class NonSpikingChemicalSynapseConv(jit.ScriptModule):
    def __init__(self, in_channels, out_channels, kernel_size, conv_dim=2, params=None, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', device=None, dtype=None, activation=ClampActivation, generator=None):
        super().__init__()
        if conv_dim == 1:
            conv = nn.Conv1d
        elif conv_dim == 2:
            conv = nn.Conv2d
        elif conv_dim == 3:
            conv = nn.Conv3d
        else:
            raise ValueError('Convolution dimension must be 1, 2, or 3')

        self.conv_left = conv(in_channels,out_channels,kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, padding_mode=padding_mode, bias=False, device=device, dtype=dtype)
        self.conv_right = conv(in_channels,out_channels,kernel_size, stride=stride, padding=padding, dilation=dilation,
                               groups=groups, padding_mode=padding_mode, bias=False, device=device, dtype=dtype)
        # remove the weights so they don't show up when calling parameters()
        shape = self.conv_right.weight.shape
        # del self.conv_left.weight
        # del self.conv_right.weight

        self.params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.randn(shape, generator=generator, dtype=dtype).to(device)),
            'reversal': nn.Parameter((2*torch.randn(shape, generator=generator, dtype=dtype)-1).to(device))
        })
        if params is not None:
            self.params.update(params)
        conductance = torch.clamp(self.params['conductance'], min=0)
        left = torch.zeros(shape, dtype=dtype, device=device)
        right = torch.zeros(shape, dtype=dtype, device=device)
        left[0,0,:,:] = (conductance * self.params['reversal']).to(device)
        right[0,0,:,:] = conductance
        self.conv_left.weight.data = nn.Parameter(left.to(device))
        self.conv_right.weight.data = nn.Parameter(right.to(device))
        self.act = activation()

    @jit.script_method
    def forward(self,x, state_post):
        # with profiler.record_function("CONV SYNAPSE"):
        x_unsqueezed = self.act(x).unsqueeze(0).unsqueeze(0)
        out = self.conv_left(x_unsqueezed) - self.conv_right(x_unsqueezed)*state_post
        return out

class NonSpikingChemicalSynapseElementwise(jit.ScriptModule):
    def __init__(self, params=None, device=None, dtype=torch.float32, generator=None, activation=ClampActivation()):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.act = activation
        self.params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.rand(1, device=device, dtype=dtype, generator=generator).to(device)),
            'reversal': nn.Parameter(2*torch.rand(1, device=device, dtype=dtype, generator=generator).to(device)-1)
        })
        if params is not None:
            self.params.update(params)

    @jit.script_method
    def forward(self, x, state_post):
        # with profiler.record_function("ELEMENTWISE SYNAPSE"):
        out = self.params['conductance']*self.act(x) * (self.params['reversal'] - state_post)
        return out
