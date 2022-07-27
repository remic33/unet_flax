"""Unet model class file"""
from flax import linen as nn
from typing import Any, Callable
from functools import partial
from jax.image import resize
import jax.numpy as jnp

ModuleDef = Any


class ConvBlock(nn.Module):
    """Basic convolution block"""
    filters: int
    training: bool
    dropout: bool = False

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.filters, (3, 3))(x)
        y = nn.BatchNorm(use_running_average=not self.training)(y)
        y = nn.relu(y)
        y = nn.Conv(self.filters, (3, 3))(y)
        y = nn.BatchNorm(use_running_average=not self.training)(y)
        y = nn.relu(y)
        if self.dropout:
            y = nn.Dropout(0.5)(x, deterministic=not self.training)
        return y


class ExpandBlock(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.filters, (3, 3))(x)
        y = nn.relu(y)
        y = nn.Conv(self.filters, (3, 3))(y)
        return nn.relu(y)


class Unet(nn.Module):
    block: Any
    expand_block: ModuleDef
    training: bool = True
    poolings: int = 4

    @nn.compact
    def __call__(self, x, train: bool = True):
        res_blocks = []
        filters = 64
        dropout = False
        for i in range(self.poolings):
            if i == self.poolings - 1:
                dropout = True
            x = self.block(filters=filters, dropout=dropout, training=self.training)(x)
            res_blocks.append(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
            filters = filters * 2
        self.block(filters=filters, dropout=True, training=self.training)(x)
        res_blocks.reverse()
        for pool in res_blocks:
            filters = filters // 2
            x = resize(x, shape=pool.shape, method='bilinear')  # Upsampling
            x = jnp.concatenate((x, pool), axis=-1)  # Concatenate
            x = self.expand_block(filters=filters)(x)
        return nn.sigmoid(x)


BasicUnet = partial(Unet, block=ConvBlock, expand_block=ExpandBlock)
