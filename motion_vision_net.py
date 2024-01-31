from snstorch import modules as m
import torch
import torch.nn as nn


class SNSBandpass(nn.Module):
    def __init__(self, shape, device=None, dtype=torch.float32):
        super().__init__()
        if device is None:
            device = 'cpu'
        self.input = m.NonSpikingLayer(shape, device=device, dtype=dtype)
        self.syn_input_fast = m.ChemicalSynapseElementwise(device=device, dtype=dtype)
        self.fast = m.NonSpikingLayer(shape, device=device, dtype=dtype)
        self.syn_input_slow = m.ChemicalSynapseElementwise(device=device, dtype=dtype)
        self.slow = m.NonSpikingLayer(shape, device=device, dtype=dtype)
        self.syn_fast_output = m.ChemicalSynapseElementwise(device=device, dtype=dtype)
        self.syn_slow_output = m.ChemicalSynapseElementwise(device=device, dtype=dtype)
        self.output = m.NonSpikingLayer(shape, device=device, dtype=dtype)

    def forward(self, x, state_input, state_fast, state_slow, state_output):
        input2fast = self.syn_input_fast(state_input, state_fast)
        input2slow = self.syn_input_slow(state_input, state_slow)
        fast2out = self.syn_fast_output(state_fast, state_output)
        slow2out = self.syn_slow_output(state_slow, state_output)

        state_input = (x, state_input)
        state_fast = self.fast(input2fast, state_fast)
        state_slow = self.slow(input2slow, state_slow)
        state_output = self.output(fast2out+slow2out, state_output)
        return state_input, state_fast, state_slow, state_output

class SNSMotionVisionEye(nn.Module):
    def __init__(self, shape_input, shape_field, reversal_ex=torch.tensor([2.0]), reversal_in=torch.tensor([-2.0]),
                 reversal_mod=torch.tensor([0.0]), device=None, dtype=torch.float32):
        super().__init__()
        if device is None:
            device = 'cpu'

        # Network
        self.reversal_ex = nn.Parameter(reversal_ex.to(device))
        self.reversal_in = nn.Parameter(reversal_in.to(device))
        self.reversal_mod = nn.Parameter(reversal_mod.to(device))

        # Retina
        self.input = m.NonSpikingLayer(shape_input, device=device, dtype=dtype)

        # Lamina
        shape_post_conv = shape_input - [shape_field-1, shape_field-1]
        # Bo
        self.syn_input_bandpass_on = m.ChemicalSynapseConv2d(1,1,shape_field)
        self.bandpass_on = SNSBandpass(shape_post_conv, device=device, dtype=dtype)
        # L
        self.syn_input_lowpass = m.ChemicalSynapseConv2d(1, 1, shape_field)
        self.lowpass = m.NonSpikingLayer(shape_post_conv, device=device, dtype=dtype)
        # Bf
        self.syn_input_bandpass_off = m.ChemicalSynapseConv2d(1, 1, shape_field)
        self.bandpass_off = SNSBandpass(shape_post_conv, device=device, dtype=dtype)

        # Medulla (On)
        # Do
        self.syn_bandpass_on_direct_on = m.ChemicalSynapseElementwise(shape_post_conv, device=device, dtype=dtype)
        self.direct_on = m.NonSpikingLayer(shape_post_conv, device=device, dtype=dtype)
        # Eo
        self.syn_lowpass_enhance_on = m.ChemicalSynapseElementwise(shape_post_conv, device=device, dtype=dtype)
        self.enhance_on = m.NonSpikingLayer(shape_post_conv, device=device, dtype=dtype)
        # So
        self.syn_direct_on_suppress_on = m.ChemicalSynapseElementwise(shape_post_conv, device=device, dtype=dtype)
        self.suppress_on = m.NonSpikingLayer(shape_post_conv, device=device, dtype=dtype)

        # Medulla (Off)
        # Df
        self.syn_bandpass_off_direct_off = m.ChemicalSynapseElementwise(shape_post_conv, device=device, dtype=dtype)
        self.syn_suppress_off_direct_off = m.ChemicalSynapseElementwise(shape_post_conv, device=device, dtype=dtype)
        self.direct_off = m.NonSpikingLayer(shape_post_conv, device=device, dtype=dtype)
        # Ef
        self.syn_lowpass_enhance_off = m.ChemicalSynapseElementwise(shape_post_conv, device=device, dtype=dtype)
        self.syn_suppress_off_enhance_off = m.ChemicalSynapseElementwise(shape_post_conv, device=device, dtype=dtype)
        self.enhance_off = m.NonSpikingLayer(shape_post_conv, device=device, dtype=dtype)
        # Sf
        self.syn_direct_off_suppress_off = m.ChemicalSynapseElementwise(shape_post_conv, device=device, dtype=dtype)
        self.syn_enhance_off_suppress_off = m.ChemicalSynapseElementwise(shape_post_conv, device=device, dtype=dtype)
        self.suppress_off = m.NonSpikingLayer(shape_post_conv, device=device, dtype=dtype)

        # Detectors
        shape_emd = shape_post_conv - [2,2]
        # On EMD
        # TODO:
        cond_enhance_on_emd = nn.Parameter()
        # CW
        self.syn_enhance_on_emd_cw_on = m.ChemicalSynapseConv2d(1,1,3, device=device, dtype=dtype)
        self.syn_direct_on_emd_cw_on = m.ChemicalSynapseConv2d(1, 1, 3, device=device, dtype=dtype)
        self.syn_suppress_on_emd_cw_on = m.ChemicalSynapseConv2d(1, 1, 3, device=device, dtype=dtype)
        self.emd_cw_on = m.NonSpikingLayer(shape_emd, device=device, dtype=dtype)
