"""Unet model class file"""
from flax import linen as nn
from typing import Any, Callable
from functools import partial
from jax.image import resize
import jax.numpy as jnp

ModuleDef = Any


class ConvBlock(nn.model):
    """Basic convolution block"""
    filters: int
    dropout: bool = False
    training: bool

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.filters, (3, 3))(x)
        y = nn.BatchNorm(use_running_average=not self.training)(y)
        y = nn.Relu()(y)
        y = nn.Conv(self.filters, (3, 3))(y)
        y = nn.BatchNorm(use_running_average=not self.training)(y)
        y = nn.Relu()(y)
        if self.dropout:
            y = nn.Dropout(0.5)
        return y


class ExpandBlock:
    dropout: bool
    filters: int

    def __call__(self, x):
        y = nn.Conv(self.filters, (3, 3))(x)
        y = nn.Relu()(y)
        y = nn.Conv(self.filters, (3, 3))(y)
        y = nn.Relu()(y)
        return y


class Unet(nn.model):
    poolings: int = 4
    block: Any
    expand_block: ModuleDef
    training: bool

    @nn.compact
    def __call__(self, x, train: bool = True):
        res_blocks = []
        filters = 64
        dropout = False
        for i in range(self.pooling):
            if i == self.pooling - 1:
                dropout = True
            x = self.block(filters=filters, dropout=dropout, training=self.training)(x)
            res_blocks.append(x)
            nn.pool()(x)
            filters = filters * 2
        self.block(filters=filters, dropout=True, training=self.training)(x)
        res_blocks.reverse()
        for pool in res_blocks:
            filters = filters / 2
            x = resize(x, shape=pool.shape, method='bilinear')  # Upsampling
            x = jnp.concatenate((x, pool), axis=-1)  # Concatenate
            x = self.expand_block(dropout=dropout, filters=filters)(x)
            dropout = False
        return nn.sigmoid()(x)


BasicUnet = partial(Unet, block=ConvBlock, expand_block=ExpandBlock)
