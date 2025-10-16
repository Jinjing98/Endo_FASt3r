from transformers import AutoImageProcessor, AutoModelForDepthEstimation, DepthAnythingForDepthEstimation
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class DoRA(nn.Module):
    def __init__(self, m: nn.Module , lora_r= 1, lora_dropout = 0.0, lora_s = 1.0, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = m.in_features
        self.out_features = m.out_features
        self.original_weight_matrix = m.weight.detach()
        self.weight_m = nn.Parameter(torch.empty((self.out_features, 1), **factory_kwargs),requires_grad=True)
        self.weight_v = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs),requires_grad=False)
        self.lora_r = lora_r
        if m.bias is not None:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs), requires_grad = True)
        else:
            self.register_parameter('bias', None)
        ### init weight_m and weight_v and bias
        with torch.no_grad():
            m = nn.utils.weight_norm(m, dim=0)
            copy_weight_m = m.weight_g.detach()
            copy_weight_v = m.weight_v.detach()
            self.weight_m.copy_(copy_weight_m)
            self.weight_v.copy_(copy_weight_v)
            if m.bias is not None:
                copy_bias = m.bias.detach()
                self.bias.copy_(copy_bias)

        self.lora_A = nn.Parameter(m.weight.new_zeros((self.lora_r, self.in_features)))
        self.lora_B = nn.Parameter(m.weight.new_zeros((self.out_features, self.lora_r)))
        self.scaling = lora_s  ## don't know if this is really needed as tining scaling is essentially the same as tuning learning rate

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.merged = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        new_weight_v = self.weight_v + (self.lora_A.T @ self.lora_B.T).T * self.scaling
        weight = ( self.weight_m / (torch.linalg.norm(new_weight_v,dim=1)).unsqueeze(1)) * (self.weight_v + (self.lora_A.T @ self.lora_B.T).T * self.scaling)

        return nn.functional.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, lora_dim={}, lora_scale={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.lora_r, self.scaling
        )


class DoRAInitializer:
    def __init__(self, model, r=[14,14,12,12,10,10,8,8,8,8,8,8], lora=['q', 'v']):
        self.model = model
        self.r = r
        self.lora = lora
        self.w_As = []
        self.w_Bs = []
        self.initialize_dora()

    def initialize_dora(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(self.model.backbone.encoder.layer):
            dim = blk.attention.attention.query.in_features

            if 'q' in self.lora:
                w_q = blk.attention.attention.query
                # w_a_linear_q = nn.Linear(dim, self.r[t_layer_i], bias=False)
                # w_b_linear_q = nn.Linear(self.r[t_layer_i], dim, bias=False)
                # self.w_As.append(w_a_linear_q)
                # self.w_Bs.append(w_b_linear_q)
                blk.attention.attention.query = DoRA(w_q, lora_r = self.r[t_layer_i])

            if 'v' in self.lora:
                w_v = blk.attention.attention.value
                # w_a_linear_v = nn.Linear(dim, self.r[t_layer_i], bias=False)
                # w_b_linear_v = nn.Linear(self.r[t_layer_i], dim, bias=False)
                # self.w_As.append(w_a_linear_v)
                # self.w_Bs.append(w_b_linear_v)
                blk.attention.attention.value = DoRA(w_v, lora_r = self.r[t_layer_i])

            if 'k' in self.lora:
                w_k = blk.attention.attention.key
                # w_a_linear_k = nn.Linear(dim, self.r[t_layer_i], bias=False)
                # w_b_linear_k = nn.Linear(self.r[t_layer_i], dim, bias=False)
                # self.w_As.append(w_a_linear_k)
                # self.w_Bs.append(w_b_linear_k)
                blk.attention.attention.key = DoRA(w_k, lora_r = self.r[t_layer_i])

        # self.reset_parameters()
        print("DoRA params initialized!")
        print("DoRA rank is:", self.r)


class DepthAnythingDepthEstimationHead(nn.Module):

    def __init__(self, model_head):
        super().__init__()

        # self.head_in_index = config.head_in_index
        # self.patch_size = config.patch_size

        # features = config.fusion_hidden_size
        self.conv1 = model_head.conv1
        self.conv2 = model_head.conv2
        self.activation1 = nn.ReLU()
        self.conv3 = model_head.conv3
        self.activation2 = nn.Sigmoid()

    def forward(self, hidden_states, height, width):
        #hidden_states = hidden_states[self.head_in_index]#[1, 64, 144, 176]
        #print('Final Head:',hidden_states.shape)
        predicted_depth = self.conv1(hidden_states)
        predicted_depth = nn.functional.interpolate(
            predicted_depth,
            (int(height), int(width)),
            mode="bilinear",
            align_corners=True,
        )
        # print('head predicted_depth:', predicted_depth.shape)
        predicted_depth = self.conv2(predicted_depth)
        predicted_depth = self.activation1(predicted_depth)
        predicted_depth = self.conv3(predicted_depth)
        predicted_depth = self.activation2(predicted_depth)
        # predicted_depth = predicted_depth.squeeze(dim=1)  # shape (batch_size, height, width)

        return predicted_depth


class Customised_DAM(nn.Module):
    def __init__(self, r = [14,14,12,12,10,10,8,8,8,8,8,8], lora = ['q', 'v']):
        super(Customised_DAM, self).__init__()
        print("DARES_MoDoRA")
        model = DepthAnythingForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.r = r
        self.lora = lora
        self.config = model.config
        self.backbone = model.backbone

        # Initialize LoRA parameters
        # self.lora_initializer = LoRAInitializer(model, r, lora)
        self.dora_initializer = DoRAInitializer(model, r, lora)

        self.neck = model.neck
        model_head = model.head
        self.head = DepthAnythingDepthEstimationHead(model_head)
        model.post_init()

    def save_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        decode_head_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.decode_head, torch.nn.DataParallel) or isinstance(self.decode_head, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.decode_head.module.state_dict()
        else:
            state_dict = self.decode_head.state_dict()
        for key, value in state_dict.items():
            decode_head_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **decode_head_tensors}
        torch.save(merged_dict, filename)

        print('saved lora parameters to %s.' % filename)

    def load_parameters(self, filename: str, device: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename, map_location=device)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        decode_head_dict = self.decode_head.state_dict()
        decode_head_keys = decode_head_dict.keys()

        # load decode head
        decode_head_keys = [k for k in decode_head_keys]
        decode_head_values = [state_dict[k] for k in decode_head_keys]
        decode_head_new_state_dict = {k: v for k, v in zip(decode_head_keys, decode_head_values)}
        decode_head_dict.update(decode_head_new_state_dict)

        self.decode_head.load_state_dict(decode_head_dict)

        print('loaded lora parameters from %s.' % filename)

    def forward(self, pixel_values):
        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=None, output_attentions=None
        ) # pixel_values:[3, 256, 320] , output:4, [1, 397, 384]
        hidden_states = outputs.feature_maps
        # print(hidden_states[0])
        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size
        # print('h,w, p, ph, pw', height, width, patch_size, patch_height, patch_width) #h,w, p, ph, pw 256 320 14 18 22
        # print('hidden_states1', len(hidden_states), hidden_states[0].shape, hidden_states[1].shape, hidden_states[2].shape, hidden_states[3].shape)
        hidden_states = self.neck(hidden_states, patch_height, patch_width)
        #[1, 64, 18, 22], [1, 64, 36, 44], [1, 64, 72, 88], [1, 64, 144, 176]
        # print('hidden_states2', len(hidden_states), hidden_states[0].shape, hidden_states[1].shape, hidden_states[2].shape, hidden_states[3].shape)
        #predicted_depth = self.head(hidden_states[3], height, width)
        outputs = {}
        outputs[("disp", 0)] = self.head(hidden_states[3], height, width)
        outputs[("disp", 1)] = self.head(hidden_states[2], height/2, width/2)
        outputs[("disp", 2)] = self.head(hidden_states[1], height/4, width/4)
        outputs[("disp", 3)] = self.head(hidden_states[0], height/8, width/8)
        # print(outputs[("disp", 0)].shape, outputs[("disp", 1)].shape,outputs[("disp", 2)].shape,outputs[("disp", 3)].shape,)
        return outputs
        #return outputs[("disp", 0)]
        # return predicted_depth


class Vector_MoRA(nn.Module):
    def __init__(self, w_qkv, mora_rank, lora_dropout):
        super().__init__()
        self.base_layer = w_qkv
        self.r = mora_rank  
        self.in_features = w_qkv.in_features
        self.out_features = w_qkv.out_features
        # LoRA dropout
        self.lora_dropout = nn.ModuleDict({
            'default': nn.Dropout(p=lora_dropout)
        })

        # LoRA A and B matrices
        # self.lora_A = nn.ModuleDict({
        #     'default': nn.Linear(self.in_features, self.r, bias=False)
        # })

        # self.lora_A = nn.ModuleDict({'default':nn.Linear(self.r, self.r, bias=False)})


        self.lora_A = nn.Linear(self.r, self.r, bias=False)
        nn.init.zeros_(self.lora_A.weight)
        #nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

    def forward(self, x):
        in_f, out_f = self.in_features, self.out_features
        result = self.base_layer(x)
        x = self.lora_dropout['default'](x) 
        #########type6 compression
        sum_inter = in_f // self.r
        rb1 = in_f//self.r if in_f % self.r == 0 else in_f//self.r + 1
        if in_f % self.r != 0:
            pad_size = self.r - in_f % self.r
            x = torch.cat([x, x[..., :pad_size]], dim=-1)
            sum_inter += 1
        in_x = x.view(*x.shape[:-1], sum_inter, self.r)
        if not hasattr(self, 'cos') and not hasattr(self, 'sin'):
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.r, 2).float() / self.r))
            t = torch.arange(rb1)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos = emb.cos().unsqueeze(0).to(x.device).to(x.dtype)
            self.sin = emb.sin().unsqueeze(0).to(x.device).to(x.dtype)
        rh_in_x = torch.cat((-in_x[..., self.r//2:], in_x[..., :self.r//2]), dim=-1)
        in_x = in_x*self.cos + rh_in_x*self.sin
        ############################

        out_x = self.lora_A(in_x)

        ###############type6 decompression
        out_x = out_x.view(*x.shape[:-1], -1)[..., :out_f]
        if out_x.shape[-1] < out_f:
            repeat_time = out_f // out_x.shape[-1]
            if out_f % out_x.shape[-1] != 0:
                repeat_time += 1
            out_x = torch.cat([out_x]*repeat_time, dim=-1)[..., :out_f]     
        # print("out_x",out_x)
        return result + out_x

class VectorMoRAInitializer:
    def __init__(self, model, mora_rank_coefficients, lora_dropout=0.01):
        self.model = model
        self.lora_dropout = lora_dropout
        self.mora_rank_coefficients = mora_rank_coefficients

    def calculate_mora_ranks(self):
        return [self.base_rank * coeff for coeff in self.mora_rank_coefficients]

    def initialize_mora(self):
        print("DAM mora rank is ", self.mora_rank_coefficients)

        # for param in self.model.backbone.parameters():
        #     param.requires_grad = False
        # print("lora frozen!!!")

        # for name, param in self.model.backbone.named_parameters():
        #   if "linear_a" in name or "linear_b" in name:
        #       # Keep LoRA offsets trainable
        #       param.requires_grad = True
        #   else:
        #       # Freeze everything else
        #       param.requires_grad = False
        # print("lora unfrozen!!!!!!")

        for t_layer_i, blk in enumerate(self.model.backbone.encoder.layer):
            w_q = blk.attention.attention.query
            w_v = blk.attention.attention.value
            # w_k = blk.attention.attention.key
            mora_rank = self.mora_rank_coefficients[t_layer_i]
            # print(f'-------- layer: {t_layer_i}, current mora rank: {mora_rank }--------')
            blk.attention.attention.query = Vector_MoRA(w_q, mora_rank, self.lora_dropout)
            blk.attention.attention.value = Vector_MoRA(w_v, mora_rank, self.lora_dropout)
            # blk.attention.attention.key = Vector_MoRA(w_k, mora_rank, self.lora_dropout)

        print("Vector MoRA params initialized!")
        return self.model



class Endo_FASt3r_depth(nn.Module):
    def __init__(self, mora_ranks=None, mora_dropout=0.01):
        """
        Create a new 'MoRA-ized' model by:
          1) Building a Customised_DAM instance (which has DoRA).
          2) Running the VectorMoRAInitializer on it to add the Vector_MoRA layers.
        """
        super().__init__()
        if mora_ranks is None:
            # Default rank coefficients
            # mora_ranks = [14,14,12,12,10,10,8,8,8,8,8,8]
            mora_ranks = [14,14,12,12,10,10,8,8,8,8,8,8]

        # 1) Build the standard Customised_DAM (which loads DepthAnything + DoRA)
        self.model_depth = Customised_DAM()

        # 2) Initialize MoRA on top of that
        self.mora_initializer = VectorMoRAInitializer(
            self.model_depth, mora_ranks, lora_dropout=mora_dropout
        )
        self.model_depth = self.mora_initializer.initialize_mora()
        print("MoDoRA in Depth model")

    def forward(self, pixel_values: torch.Tensor):
        """
        Forward pass that just calls the underlying model_depth forward.
        If you want to do extra processing, you can do it here.
        """
        return self.model_depth(pixel_values)

if __name__ == "__main__":
    model = Endo_FASt3r_depth()
    print(model)
    pixel_values = torch.randn(1, 3, 256, 320)
    outputs = model(pixel_values)
    for key, value in outputs.items():
        print(key, value.shape)