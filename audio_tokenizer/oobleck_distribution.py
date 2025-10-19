# Extracted from diffusers.models.autoencoders.autoencoder_oobleck
# To avoid dependency issues with diffusers and transformers version compatibility

from typing import Optional
import torch
import torch.nn as nn


def randn_tensor(
    shape,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """A helper function to create random tensors with the desired `generator`."""
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)

    # Handle generator properly
    if device is not None and device.type == "cpu":
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
    else:
        # For CUDA, we need to be careful with generators
        latents = torch.randn(shape, generator=generator, device="cpu", dtype=dtype)
        return latents.to(device)


class OobleckDiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.scale = parameters.chunk(2, dim=1)
        self.std = nn.functional.softplus(self.scale) + 1e-4
        self.var = self.std * self.std
        self.logvar = torch.log(self.var)
        self.deterministic = deterministic

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "OobleckDiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return (self.mean * self.mean + self.var - self.logvar - 1.0).sum(1).mean()
            else:
                normalized_diff = torch.pow(self.mean - other.mean, 2) / other.var
                var_ratio = self.var / other.var
                logvar_diff = self.logvar - other.logvar

                kl = normalized_diff + var_ratio + logvar_diff - 1

                kl = kl.sum(1).mean()
                return kl

    def mode(self) -> torch.Tensor:
        return self.mean