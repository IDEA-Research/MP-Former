from collections import OrderedDict

import torch
import numpy as np

def slprint(x, name='x'):
    if isinstance(x, (torch.Tensor, np.ndarray)):
        print(f'{name}.shape:', x.shape)
    elif isinstance(x, (tuple, list)):
        print('type x:', type(x))
        for i in range(min(10, len(x))):
            slprint(x[i], f'{name}[{i}]')
    elif isinstance(x, dict):
        for k,v in x.items():
            slprint(v, f'{name}[{k}]')
    else:
        print(f'{name}.type:', type(x))

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict

def renorm(img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        -> torch.FloatTensor:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.dim() == 3 or img.dim() == 4, "img.dim() should be 3 or 4 but %d" % img.dim() 
    if img.dim() == 3:
        assert img.size(0) == 3, 'img.size(0) shoule be 3 but "%d". (%s)' % (img.size(0), str(img.size()))
        img_perm = img.permute(1,2,0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2,0,1)
    else: # img.dim() == 4
        assert img.size(1) == 3, 'img.size(1) shoule be 3 but "%d". (%s)' % (img.size(1), str(img.size()))
        img_perm = img.permute(0,2,3,1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0,3,1,2)



class CocoClassMapper():
    def __init__(self) -> None:
        self.category_map_str = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}
        self.origin2compact_mapper = {int(k):v-1 for k,v in self.category_map_str.items()}
        self.compact2origin_mapper = {int(v-1):int(k) for k,v in self.category_map_str.items()}

    def origin2compact(self, idx):
        return self.origin2compact_mapper[int(idx)]

    def compact2origin(self, idx):
        return self.compact2origin_mapper[int(idx)]

def to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, list):
        return [to_device(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: to_device(v, device) for k,v in item.items()}
    else:
        raise NotImplementedError("Call Shilong if you use other containers! type: {}".format(type(item)))



# 
def get_gaussian_mean(x, axis, other_axis, softmax=True):
    """

    Args:
        x (float): Input images(BxCxHxW)
        axis (int): The index for weighted mean
        other_axis (int): The other index

    Returns: weighted index for axis, BxC

    """
    mat2line = torch.sum(x, axis=other_axis)
    # mat2line = mat2line / mat2line.mean() * 10
    if softmax:
        u = torch.softmax(mat2line, axis=2)
    else:
        u = mat2line / (mat2line.sum(2, keepdim=True) + 1e-6)
    size = x.shape[axis]
    ind = torch.linspace(0, 1, size).to(x.device)
    batch = x.shape[0]
    channel = x.shape[1]
    index = ind.repeat([batch, channel, 1])
    mean_position = torch.sum(index * u, dim=2)
    return mean_position

def get_expected_points_from_map(hm, softmax=True):
    """get_gaussian_map_from_points
        B,C,H,W -> B,N,2 float(0, 1) float(0, 1)
        softargmax function

    Args:
        hm (float): Input images(BxCxHxW)

    Returns: 
        weighted index for axis, BxCx2. float between 0 and 1.

    """
    # hm = 10*hm
    B,C,H,W = hm.shape
    y_mean = get_gaussian_mean(hm, 2, 3, softmax=softmax) # B,C
    x_mean = get_gaussian_mean(hm, 3, 2, softmax=softmax) # B,C
    # return torch.cat((x_mean.unsqueeze(-1), y_mean.unsqueeze(-1)), 2)
    return torch.stack([x_mean, y_mean], dim=2)

# Positional encoding (section 5.1)
# borrow from nerf
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    import torch.nn as nn
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class APOPMeter():
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, pred, gt):
        """
        Input:
            pred, gt: Tensor()
        """
        assert pred.shape == gt.shape
        self.tp += torch.logical_and(pred == 1, gt == 1).sum().item()
        self.fp += torch.logical_and(pred == 1, gt == 0).sum().item()
        self.tn += torch.logical_and(pred == 0, gt == 0).sum().item()
        self.tn += torch.logical_and(pred == 1, gt == 0).sum().item()

    def update_cm(self, tp, fp, tn, fn):
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.tn += fn

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

import argparse
from .slconfig import SLConfig
def get_raw_dict(args):
    """
    return the dicf contained in args.
    
    e.g:
        >>> with open(path, 'w') as f:
                json.dump(get_raw_dict(args), f, indent=2)
    """
    if isinstance(args, argparse.Namespace): 
        return vars(args)   
    elif isinstance(args, dict):
        return args
    elif isinstance(args, SLConfig):
        return args._cfg_dict
    else:
        raise NotImplementedError("Unknown type {}".format(type(args)))


def stat_tensors(tensor):
    assert tensor.dim() == 1
    tensor_sm = tensor.softmax(0)
    entropy = (tensor_sm * torch.log(tensor_sm + 1e-9)).sum()

    return {
        'max': tensor.max(),
        'min': tensor.min(),
        'mean': tensor.mean(),
        'var': tensor.var(),
        'std': tensor.var() ** 0.5,
        'entropy': entropy
    }