import torch
import numpy as np

from torchtyping import TensorType, patch_typeguard  # type: ignore
from typeguard import typechecked
from typing import Callable, Union

patch_typeguard()  # use before @typechecked

def build_rff_projector_function(in_dim: int, out_dim: int, width:float) -> Callable:
	weights = torch.normal(mean=torch.zeros(in_dim, out_dim), std=width*torch.ones(in_dim, out_dim))
	biases	= 2*np.pi*torch.rand(out_dim)
	@typechecked
	# def rff(x: TensorType["batch": ..., "in_dim": in_dim]) -> TensorType["batch": ..., "out_dim": out_dim]:
	def rff(x: TensorType[..., in_dim]) -> TensorType[..., out_dim]:
		linear_out 	= torch.einsum("...i, io -> ...o", x, weights) + biases 
		rff_vec		= torch.cos(linear_out)/out_dim**.5

		return rff_vec/rff_vec.norm()

	return rff

in_dim  = 3
out_dim = 100
width 	= 0.5
rff = build_rff_projector_function(in_dim, out_dim, width)
x = torch.normal(torch.zeros(in_dim))
w = torch.normal(torch.zeros(out_dim))

print(rff(x)@rff(x))
print(w@rff(x))