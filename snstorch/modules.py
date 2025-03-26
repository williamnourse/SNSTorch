import torch
import torch.nn as nn
import torch.autograd.profiler as profiler
import torch.jit as jit
from typing import List

class ATan(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function. From snnTorch
    """

    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            ctx.alpha
            / 2
            / (1 + (torch.pi / 2 * ctx.alpha * input_).pow_(2))
            * grad_input
        )
        return grad, None


def atan(alpha=2.0):
    """ArcTan surrogate gradient enclosed with a parameterized slope."""
    alpha = alpha

    def inner(x):
        return ATan.apply(x, alpha)

    return inner

class NonSpikingLayer(nn.Module):
    def __init__(self, size, params=None, generator=None, device=None, dtype=torch.float32,
                 ):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.params = nn.ParameterDict({
            'tau': nn.Parameter(0.5*torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'leak': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'rest': nn.Parameter(torch.zeros(size, dtype=dtype).to(device)),
            'bias': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'init': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device))
        })
        if params is not None:
            self.params.update(params)

    # @jit.script_method
    def forward(self, x, state):
        # if state is None:
        #     state = self.params['init']
        # with profiler.record_function("NEURAL UPDATE"):
        state = state + self.params['tau'] * (-self.params['leak'] * (state - self.params['rest']) + self.params['bias'] + x)
            # state += new_state
        # if x is not None:
        #     state += self.params['tau']*x
        return state

class TraceLayer(nn.Module):
    def __init__(self, size, params=None, generator=None, device=None, dtype=torch.float32):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.params = nn.ParameterDict({
            'tau': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device))
        })
        if params is not None:
            self.params.update(params)

    def forward(self, x, state):
        # if state is None:
        #     state = self.params['init']
        # with profiler.record_function("NEURAL UPDATE"):
        state = state * (1 - self.params['tau']) + x
            # state += new_state
        # if x is not None:
        #     state += self.params['tau']*x
        return state

class AdaptiveNonSpikingLayer(nn.Module):
    def __init__(self, size, params=None, generator=None, device=None, dtype=torch.float32,
                 ):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.params = nn.ParameterDict({
            'tauU': nn.Parameter(0.5*torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'leakU': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'restU': nn.Parameter(torch.zeros(size, dtype=dtype).to(device)),
            'biasU': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'initU': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'tauA': nn.Parameter(0.5*torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'leakA': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'restA': nn.Parameter(torch.zeros(size, dtype=dtype).to(device)),
            'gainA': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'initA': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device))
        })
        if params is not None:
            self.params.update(params)

    # @jit.script_method
    def forward(self, x, state, adapt):
        # if state is None:
        #     state = self.params['init']
        # with profiler.record_function("NEURAL UPDATE"):
        state_new = state + self.params['tauU'] * (-self.params['leakU'] * (state - self.params['restU']) + self.params['biasU'] + x - self.params['gainA']*adapt)
        adapt_new = adapt + self.params['tauA'] * (-self.params['leakA'] * (adapt - self.params['restA']) + state)
            # state += new_state
        # if x is not None:
        #     state += self.params['tau']*x
        return state_new, adapt_new

class SpikingLayer(nn.Module):
    def __init__(self, size, params=None, generator=None, device=None, dtype=torch.float32, surrogate=None
                 ):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.params = nn.ParameterDict({
            'tauU': nn.Parameter(0.5*torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'leakU': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'restU': nn.Parameter(torch.zeros(size, dtype=dtype).to(device)),
            'biasU': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'initU': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'theta': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'resetU': nn.Parameter(torch.zeros(size, dtype=dtype).to(device)),
        })
        if params is not None:
            self.params.update(params)
        if surrogate is None:
            surrogate = atan()
        self.spike_grad = surrogate

    def fire(self, state):
        state_shift = state - self.params['theta']
        spikes = self.spike_grad(state_shift)
        return spikes

    # @jit.script_method
    def forward(self, x, state):
        # if state is None:
        #     state = self.params['init']
        # with profiler.record_function("NEURAL UPDATE"):
        state_new = state + self.params['tauU'] * (-self.params['leakU'] * (state - self.params['restU']) + self.params['biasU'] + x)
        spikes = self.fire(state_new)
        spikes_mask = -1*(spikes-1)
        state_new = state_new*spikes_mask + self.params['resetU']*spikes
            # state += new_state
        # if x is not None:
        #     state += self.params['tau']*x
        return spikes, state_new

class AdaptiveSpikingLayer(nn.Module):
    def __init__(self, size, params=None, generator=None, device=None, dtype=torch.float32, surrogate=None
                 ):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.params = nn.ParameterDict({
            'tauU': nn.Parameter(0.5*torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'leakU': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'restU': nn.Parameter(torch.zeros(size, dtype=dtype).to(device)),
            'biasU': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'initU': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'resetU': nn.Parameter(torch.zeros(size, dtype=dtype).to(device)),
            'tauTheta': nn.Parameter(0.5 * torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'leakTheta': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'm': nn.Parameter(torch.zeros(size, dtype=dtype).to(device)),
            'initTheta': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
            'incTheta': nn.Parameter(torch.rand(size, dtype=dtype, generator=generator).to(device)),
        })
        if params is not None:
            self.params.update(params)
        if surrogate is None:
            surrogate = atan()
        self.spike_grad = surrogate

    def fire(self, state, theta):
        state_shift = state - theta
        spikes = self.spike_grad(state_shift)
        return spikes

    # @jit.script_method
    def forward(self, x, state, theta):
        # if state is None:
        #     state = self.params['init']
        # with profiler.record_function("NEURAL UPDATE"):
        state_new = state + self.params['tauU'] * (-self.params['leakU'] * (state - self.params['restU']) + self.params['biasU'] + x)
        theta_new = theta + self.params['tauTheta'] * (self.params['leakTheta'] * (self.params['initTheta'] - theta) +
                                                       self.params['m']*(state - self.params['restU']))
        spikes = self.fire(state_new, theta_new)
        spikes_mask = -1*(spikes-1)
        state_new = state_new*spikes_mask + self.params['resetU']*spikes
        theta_new = torch.clamp(theta_new + self.params['incTheta']*spikes,min=0)
            # state += new_state
        # if x is not None:
        #     state += self.params['tau']*x
        return spikes, state_new, theta_new


class ClampActivation(nn.Module):
    def __init__(self):
        super().__init__()

    # @jit.script_method
    def forward(self, x):
        return torch.clamp(x,0,1)

class PiecewiseActivation(nn.Module):
    def __init__(self, min_val=0, max_val=1):
        super().__init__()
        self.min_val = min_val
        self.inv_range = 1/(max_val-min_val)

    # @jit.script_method
    def forward(self,x):
        # with profiler.record_function("PIECEWISE SIGMOID"):
            # x -= self.min_val
            # x *= self.inv_range
            # x.clamp_(0,1)
        return torch.clamp((x - self.min_val) * self.inv_range, 0, 1)
        # return x
        # return torch.clamp((x-self.min_val)/self.range,0,1)


class NonSpikingChemicalSynapseLinear(nn.Module):
    def __init__(self, size_pre, size_post, params=None, activation=ClampActivation, device=None,
                 dtype=torch.float32, generator=None):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.params = nn.ParameterDict({
            'conductance': nn.Parameter((1/size_pre)*torch.rand([size_post,size_pre],dtype=dtype, generator=generator).to(device)),
            'reversal': nn.Parameter((4*torch.rand([size_post,size_pre], generator=generator)-2).to(device))
        })
        if params is not None:
            self.params.update(params)
        self.activation = activation()

    # @jit.script_method
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

    def setup(self):
        self.params['conductance'] = torch.clamp(self.params['conductance'], min=0)

class SpikingChemicalSynapseLinear(nn.Module):
    def __init__(self, size_pre, size_post, params=None, device=None,
                 dtype=torch.float32, generator=None):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.params = nn.ParameterDict({
            'tau': nn.Parameter(0.5 * torch.rand([size_post,size_pre], dtype=dtype, generator=generator).to(device)),
            'conductanceInc': nn.Parameter((1/size_pre)*torch.rand([size_post,size_pre],dtype=dtype, generator=generator).to(device)),
            'reversal': nn.Parameter((4*torch.rand([size_post,size_pre], generator=generator)-2).to(device))
        })
        if params is not None:
            self.params.update(params)

    # @jit.script_method
    def forward(self, spikes_pre, state_syn, state_post):
        state_syn_new = state_syn * (1 - self.params['tau'])
        if state_syn.dim()>2:
            left = torch.sum(state_syn_new * self.params['reversal'], dim=2)
            right = state_post * torch.sum(state_syn_new, dim=2)
        else:
            left = torch.sum(state_syn_new * self.params['reversal'],dim=1)
            right = state_post*torch.sum(state_syn_new,dim=1)
        out = left-right
        state_syn_new = state_syn_new + spikes_pre*self.params['conductanceInc']
        return out, state_syn_new

    def setup(self):
        self.params['conductance'] = torch.clamp(self.params['conductance'], min=0)
        self.params['tau'] = torch.clamp(self.params['tau'], min=0)

# TODO: Can this even be spiking?
class NonSpikingChemicalSynapseConv(nn.Module):
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
        if device is None:
            device = 'cpu'
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        self.device = device
        self.conv_left = conv(in_channels,out_channels,kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, padding_mode=padding_mode, bias=False, device=device, dtype=dtype)
        self.conv_right = conv(in_channels,out_channels,kernel_size, stride=stride, padding=padding, dilation=dilation,
                               groups=groups, padding_mode=padding_mode, bias=False, device=device, dtype=dtype)
        # remove the weights so they don't show up when calling parameters()
        shape = self.conv_right.weight.shape
        shape_flat = len(self.conv_right.weight.flatten())
        # del self.conv_left.weight
        # del self.conv_right.weight

        self.params = nn.ParameterDict({
            'conductance': nn.Parameter(1/shape_flat*torch.randn(shape, generator=generator, dtype=dtype).to(device)),
            'reversal': nn.Parameter((2*torch.randn(shape, generator=generator, dtype=dtype)-1).to(device))
        })
        if params is not None:
            self.params.update(params)
        self.setup()

        self.act = activation()

    # @jit.script_method
    def forward(self,x, state_post):
        # with profiler.record_function("CONV SYNAPSE"):
        if len(x.shape)==2:
            x_unsqueezed = self.act(x).unsqueeze(0).unsqueeze(0)
        elif len(x.shape)==3:
            x_unsqueezed = self.act(x).unsqueeze(0)
            x_unsqueezed = x_unsqueezed.permute(1,0,2,3)
        else:
            x_unsqueezed = self.act(x)
            # x_unsqueezed = x_unsqueezed.permute(1,0,2,3)
        out = self.conv_left(x_unsqueezed).squeeze() - self.conv_right(x_unsqueezed).squeeze()*state_post
        return out

    def setup(self):
        conductance = torch.clamp(self.params['conductance'], min=0)
        shape = self.conv_right.weight.shape
        left = torch.zeros(shape, dtype=self.dtype, device=self.device)
        right = torch.zeros(shape, dtype=self.dtype, device=self.device)
        left[0,0,:,:] = (conductance * self.params['reversal']).to(self.device)
        right[0,0,:,:] = conductance
        self.conv_left.weight.data = nn.Parameter(left.to(self.device), requires_grad=False)
        self.conv_right.weight.data = nn.Parameter(right.to(self.device), requires_grad=False)

class NonSpikingChemicalSynapseElementwise(nn.Module):
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

    # @jit.script_method
    def forward(self, x, state_post):
        # with profiler.record_function("ELEMENTWISE SYNAPSE"):
        out = self.params['conductance']*self.act(x) * (self.params['reversal'] - state_post)
        return out

    def setup(self):
        self.params['conductance'] = torch.clamp(self.params['conductance'], min=0)

class SpikingChemicalSynapseElementwise(nn.Module):
    def __init__(self, size, params=None, device=None, dtype=torch.float32, generator=None, activation=ClampActivation()):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.params = nn.ParameterDict({
            'tau': nn.Parameter(0.5 * torch.rand(1, dtype=dtype, generator=generator).to(device)),
            'conductanceInc': nn.Parameter(torch.rand(1, device=device, dtype=dtype, generator=generator).to(device)),
            'reversal': nn.Parameter(2*torch.rand(1, device=device, dtype=dtype, generator=generator).to(device)-1)
        })
        if params is not None:
            self.params.update(params)

    # @jit.script_method
    def forward(self, spikes_pre, state_syn, state_post):
        # with profiler.record_function("ELEMENTWISE SYNAPSE"):
        state_syn_new = state_syn * (1 - self.params['tau'])
        out = state_syn_new * (self.params['reversal'] - state_post)
        state_syn_new = state_syn_new + spikes_pre * self.params['conductanceInc']
        return out, state_syn_new

    def setup(self):
        self.params['conductance'] = torch.clamp(self.params['conductance'], min=0)
        self.params['tau'] = torch.clamp(self.params['tau'], min=0)
