import functools
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from models.networks import LayerNormMLP, default_init


class DrQv2Encoder(nn.Module):
    num_features: int = 32
    hidden_dims: Sequence[int] = (512,)
    activation: callable = nn.relu
    kernel_init: callable = nn.initializers.xavier_uniform()
    normalize_pixels: bool = True

    def setup(self):
        self.conv1 = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            kernel_init=self.kernel_init,
        )
        self.conv2 = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=self.kernel_init,
        )
        self.conv3 = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=self.kernel_init,
        )
        self.conv4 = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=self.kernel_init,
        )
        self.mlp = LayerNormMLP(
            hidden_dims=self.hidden_dims,
            activations=self.activation,
            activate_final=True,
            kernel_init=default_init(),
            layer_norm=True,
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32)
        if self.normalize_pixels:
            x = x / 255.0

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.reshape((*x.shape[:-3], -1))
        return self.mlp(x)


class ResnetBlock(nn.Module):
    num_features: int
    activation: callable = nn.relu
    kernel_init: callable = nn.initializers.xavier_uniform()
    batch_norm: bool = False

    def normalize(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        if not self.batch_norm:
            return x
        return nn.BatchNorm(use_running_average=not train)(x)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        if residual.shape[-1] != self.num_features:
            residual = nn.Conv(
                features=self.num_features,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                kernel_init=self.kernel_init,
            )(residual)

        y = self.activation(x)
        y = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=self.kernel_init,
        )(y)
        y = self.normalize(y, train)

        y = self.activation(y)
        y = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=self.kernel_init,
        )(y)
        y = self.normalize(y, train)
        return residual + y


class ImageResnetBlock(nn.Module):
    num_features: int = 32
    num_blocks: int = 2
    activation: callable = nn.relu
    kernel_init: callable = nn.initializers.xavier_uniform()
    max_pool: bool = True
    batch_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_init=self.kernel_init,
        )(x)

        if self.max_pool:
            x = nn.max_pool(
                x,
                window_shape=(3, 3),
                strides=(2, 2),
                padding="SAME",
            )

        for _ in range(self.num_blocks):
            x = ResnetBlock(
                num_features=self.num_features,
                activation=self.activation,
                kernel_init=self.kernel_init,
                batch_norm=self.batch_norm,
            )(x, train=train)

        return x


class ImageResnetEncoder(nn.Module):
    stack_sizes: Sequence[int] = (16, 32, 32)
    num_blocks: int = 2
    hidden_dims: Sequence[int] = (512,)
    activation: callable = nn.relu
    kernel_init: callable = nn.initializers.xavier_uniform()
    batch_norm: bool = False
    layer_norm: bool = False
    scale_pixels: bool = True

    def setup(self):
        self.image_blocks = [
            ImageResnetBlock(
                num_features=num_features,
                num_blocks=self.num_blocks,
                activation=self.activation,
                kernel_init=self.kernel_init,
                max_pool=True,
                batch_norm=self.batch_norm,
            )
            for num_features in self.stack_sizes
        ]
        self.mlp = LayerNormMLP(
            hidden_dims=self.hidden_dims,
            activations=self.activation,
            activate_final=True,
            kernel_init=default_init(),
            layer_norm=self.layer_norm,
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = x.astype(jnp.float32)
        if self.scale_pixels:
            x = x / 255.0

        for image_block in self.image_blocks:
            x = image_block(x, train=train)

        x = self.activation(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        x = x.reshape((*x.shape[:-3], -1))
        return self.mlp(x)


encoder_modules = {
    "drqv2": DrQv2Encoder,
    "resnet": ImageResnetEncoder,
    "impala_small": functools.partial(ImageResnetEncoder, num_blocks=1),
}


def make_encoder(name: str | None, **kwargs) -> nn.Module | None:
    if name is None:
        return None
    if name not in encoder_modules:
        raise ValueError(f"Unknown encoder: {name}. Available encoders: {sorted(encoder_modules)}")
    return encoder_modules[name](**kwargs)
