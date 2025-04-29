# Inherit from Function
import torch
from torch import nn
from torch.autograd import Function


class ChannelFunction(Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, snr):
        signal_power = torch.mean(input ** 2, dim=-1, keepdim=True)

        noise_power = signal_power / (10 ** (snr / 10))
        std = torch.sqrt(noise_power)
        noise = torch.randn_like(input) * std

        return input + noise

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, snr = inputs
        ctx.snr = snr
        ctx.save_for_backward(input, torch.tensor(snr, device=input.device))

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, snr = ctx.saved_tensors, ctx.snr
        grad_input = grad_weight = grad_bias = None

        signal_power = torch.mean(grad_output ** 2, dim=-1, keepdim=True)

        noise_power = signal_power / (10 ** (snr / 10))
        std = torch.sqrt(noise_power)
        noise = torch.randn_like(grad_output) * std

        # return grad_input, grad_weight, grad_bias
        return grad_output + noise, None


class AWGN(nn.Module):
    def __init__(self, snr):
        super().__init__()
        self.snr = snr
        self.total_communication = 0

    def forward(self, input):

        # See the autograd section for explanation of what happens here.
        return ChannelFunction.apply(input, self.snr)

