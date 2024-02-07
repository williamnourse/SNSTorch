import torch
import torch.nn as nn

class NonSpikingLayer(nn.Module):
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

    def forward(self,x=None,state_prev=None):
        if state_prev is None:
            state_prev = self.params['init']
        state = state_prev + self.params['tau'] * (-self.params['leak'] * (state_prev - self.params['rest']) + self.params['bias'])
        if x is not None:
            state += self.params['tau']*x
        return state

class PiecewiseActivation(nn.Module):
    def __init__(self, min_val=0, max_val=1):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self,x):
        return torch.clamp((x-self.min_val)/(self.max_val-self.min_val),0,1)

class NonSpikingChemicalSynapseLinear(nn.Module):
    def __init__(self, size_pre, size_post, params=None, activation=PiecewiseActivation, device=None,
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

    def forward(self, states):
        activated_pre = self.activation(states[0])
        if states[0].dim() > 1:
            conductance = torch.clamp(self.params['conductance'], min=0.0) * activated_pre.unsqueeze(1)
        else:
            conductance = torch.clamp(self.params['conductance'], min=0.0) * activated_pre
        if conductance.dim()>2:
            left = torch.sum(conductance * self.params['reversal'], dim=2)
            right = states[1] * torch.sum(conductance, dim=2)
        else:
            left = torch.sum(conductance * self.params['reversal'],dim=1)
            right = states[1]*torch.sum(conductance,dim=1)
        out = left-right
        return out

class NonSpikingChemicalSynapseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv_dim=2, params=None, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', device=None, dtype=None, activation=PiecewiseActivation, generator=None):
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

    def forward(self,x, state_post):
        x = self.act(x)
        # if self.training:
        #     self.conv_left.weight = (torch.clamp(self.params['conductance'], min=0) * self.params['reversal'])
        #     self.conv_right.weight = torch.clamp(self.params['conductance'], min=0)
        # print(self.conv_left(x).shape)
        out = self.conv_left(x.unsqueeze(0).unsqueeze(0)) - self.conv_right(x.unsqueeze(0).unsqueeze(0))*state_post
        return out

class NonSpikingChemicalSynapseElementwise(nn.Module):
    def __init__(self, params=None, device=None, dtype=torch.float32, generator=None, activation=PiecewiseActivation()):
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

    def forward(self, x, state_post):
        return self.params['conductance']*self.act(x) * (self.params['reversal'] - state_post)
