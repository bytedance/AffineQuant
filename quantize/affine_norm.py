import torch
import torch.nn as nn


'''
Modify normalization layer to adapt the training of learnable equivalent transformation
'''



class AffineLayerNorm(nn.Module):
    def __init__(self, ori_layer_norm) -> None:
        super().__init__()
        self.use_act_quant = True
        self.register_buffer('weight',ori_layer_norm.weight)
        if ori_layer_norm.bias is not None:
            self.register_buffer('bias',ori_layer_norm.bias)
        else:
            self.bias = None
        self.eps = ori_layer_norm.eps
        self.norm_func = nn.functional.layer_norm
        self.normalized_shape = ori_layer_norm.normalized_shape
        self.use_temporary_parameter = False
        self.register_buffer('c_inverse',None)

    def forward(self, x):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias
        # import pdb;pdb.set_trace()
        out = self.norm_func(x,self.normalized_shape,weight, bias,eps=self.eps)
        if self.c_inverse is not None and out.dtype is torch.float64:
            out = out.matmul(self.c_inverse)
        elif self.c_inverse is not None and out.dtype is torch.float16:
            out = out.type_as(self.c_inverse).matmul(self.c_inverse).half()
        elif self.c_inverse is not None and out.dtype is torch.bfloat16:
            out = out.type_as(self.c_inverse).matmul(self.c_inverse).bfloat16()
        elif self.c_inverse is not None and out.dtype is torch.float32:
            out = out.matmul(self.c_inverse)
        return out

    def set_quant_state(self, use_weight_quant, use_act_quant):
        self.use_act_quant = use_act_quant


class AffineLlamaRMSNorm(nn.Module):
    def __init__(self, ori_norm, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.register_buffer('weight',ori_norm.weight)
        self.bias = None
        self.variance_epsilon = eps
        self.use_temporary_parameter = False
        self.register_buffer('c_inverse',None)


    def forward(self, hidden_states):
        # import pdb;pdb.set_trace()
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float64 if input_dtype == torch.float64 else torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias if hasattr(self, 'bias') else None

        if bias is not None:
            out = (weight * hidden_states+bias).to(input_dtype)
        else:
            out = (weight * hidden_states).to(input_dtype)

        if self.c_inverse is not None and out.dtype is torch.float64:
            out = out.matmul(self.c_inverse)
        elif self.c_inverse is not None and out.dtype is torch.float16:
            out = out.type_as(self.c_inverse).matmul(self.c_inverse).half()
        elif self.c_inverse is not None and out.dtype is torch.bfloat16:
            out = out.type_as(self.c_inverse).matmul(self.c_inverse).bfloat16()
        elif self.c_inverse is not None and out.dtype is torch.float32:
            out = out.matmul(self.c_inverse)
        return out

