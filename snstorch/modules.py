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
        del self.conv_left.weight
        del self.conv_right.weight

        self.params = nn.ParameterDict({
            'conductance': nn.Parameter(torch.randn(shape, generator=generator, dtype=dtype).to(device)),
            'reversal': nn.Parameter((2*torch.randn(shape, generator=generator, dtype=dtype)-1).to(device))
        })
        if params is not None:
            self.params.update(params)
        self.conv_left.weight = (torch.clamp(self.params['conductance'], min=0) * self.params['reversal']).to(device)
        self.conv_right.weight = torch.clamp(self.params['conductance'], min=0)
        self.act = activation()

    def forward(self,x, state_post):
        x = self.act(x)
        if self.training:
            self.conv_left.weight = (torch.clamp(self.params['conductance'], min=0) * self.params['reversal'])
            self.conv_right.weight = torch.clamp(self.params['conductance'], min=0)
        # print(self.conv_left(x).shape)
        out = self.conv_left(x) - self.conv_right(x)*state_post
        return out

class NonSpikingChemicalSynapseElementwise(nn.Module):
    def __init__(self, conductance=None, reversal=None, device=None, dtype=torch.float32, activation=PiecewiseActivation):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.act = activation()
        if conductance is None:
            conductance = torch.rand(1, device=device, dtype=dtype)
        self.conductance = nn.Parameter(conductance.to(device), requires_grad=True)
        if reversal is None:
            reversal = torch.rand(1, device=device, dtype=dtype)
        self.reversal = nn.Parameter(reversal.to(device), requires_grad=True)

    def forward(self, x, state_post):
        return self.conductance*self.act(x) * (self.reversal - state_post)

# # Example usage
# rows, cols = 2, 2
# model = NonSpikingLayer(rows * cols)
# conductance = NonSpikingConductance(4,2)
# synapse = ChemicalSynapseLinear(4, 2)
#
# # Example single input
# input_data_single = torch.randn([rows * cols])
# output_single = model(input_data_single)
# conductance_single = conductance(output_single)
# synapse_output_single = synapse(conductance_single, model.state_0[:2])
# print(synapse_output_single)
#
# # Example batch input
# input_data_batch = torch.randn([3, rows * cols])
# output_batch = model(input_data_batch)
# state_batch = model.state_0[:2].unsqueeze(0).repeat(3,1)
# conductance_batch = conductance(output_batch)
# synapse_output_batch = synapse(conductance_batch, state_batch)
# print(synapse_output_batch)

# kernel_size = [3,3]
# layer_shape_0 = [1000,1000]
# layer_shape_1 = [layer_shape_0[i]-kernel_size[i]+1 for i in range(len(kernel_size))]
# input_data = torch.randn(layer_shape_0).unsqueeze(0)
# neurons_0 = NonSpikingLayer(layer_shape_0)
# # act = PiecewiseActivation()
# neurons_1 = NonSpikingLayer(layer_shape_1)
# conv = ChemicalSynapseConv2d(1,1,kernel_size)
#
# print('Input ',input_data)
# state = neurons_0(input_data)
# print('0 ',state)
# state = conv(state,neurons_1.state_0)
# print('1',neurons_1.state_0)
# print('Convolution Result ', state)
# print(conv.conv_left.weight)
# print(conv.conv_right.weight)
# print(conv.kernel_reversal)
# print(conv.kernel_conductance)
# print('Done')

# device = 'cuda'
# class SNSCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv0 = ChemicalSynapseConv2d(in_channels=1,
#                                       out_channels=16,
#                                       kernel_size=5,
#                                       stride=1,
#                                       padding=2)
#         self.layer0 = NonSpikingLayer([26, 26])
#         self.conv1 = ChemicalSynapseConv2d(in_channels=1,
#                                       out_channels=16,
#                                       kernel_size=5,
#                                       stride=1,
#                                       padding=2)
#         self.layer1 = NonSpikingLayer([24, 24])
#         self.out = nn.Sequential(PiecewiseActivation(),
#                                  nn.LazyLinear(10))
#
#     def forward(self, x):
#         x = self.conv0(x, self.layer0.state_0)
#         x = self.layer0(x)
#         x = self.conv1(x, self.layer1.state_0)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
#         # print(x.shape)
#         x = self.layer1(x)
#         output = self.out(x)
#         return output, x  # return x for visualization
#
#
# sns_cnn = SNSCNN().to(device)
# print(sns_cnn)
