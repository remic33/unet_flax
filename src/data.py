"""Data generator main file"""
# TODO : add elastic deformation
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

import jax
import jax.numpy as jnp
from jax import random
from jax import make_jaxpr
from jax.config import config
from jax import value_and_grad
from jax import grad, vmap, pmap, jit

import optax
from flax import linen as nn
from flax.training import train_state


def rotate_90(img):
    """Rotates an image by 90 degress k times."""
    return jnp.rot90(img, k=1, axes=(0, 1))


def identity(img):
    """Returns an image as it is."""
    return img


def flip_left_right(img):
    """Flips an image left/right direction."""
    return jnp.fliplr(img)


def flip_up_down(img):
    """Flips an image in up/down direction."""
    return jnp.flipud(img)


def random_rotate(img, rotate):
    """Randomly rotate an image by 90 degrees.

    Args:
        img: Array representing the image
        rotate: Boolean for rotating or not
    Returns:
        Rotated or an identity image
    """

    return jax.lax.cond(rotate, rotate_90, identity, img)


def random_horizontal_flip(img, flip):
    """Randomly flip an image vertically.

    Args:
        img: Array representing the image
        flip: Boolean for flipping or not
    Returns:
        Flipped or an identity image
    """

    return jax.lax.cond(flip, flip_left_right, identity, img)


def random_vertical_flip(img, flip):
    """Randomly flip an image vertically.

    Args:
        img: Array representing the image
        flip: Boolean for flipping or not
    Returns:
        Flipped or an identity image
    """

    return jax.lax.cond(flip, flip_up_down, identity, img)


# All the above function are written to work on a single example.
# We will use `vmap` to get a version of these functions that can
# operate on a batch of images. We will also add the `jit` transformation
# on top of it so that the whole pipeline can be compiled and executed faster
random_rotate_jitted = jit(vmap(random_rotate, in_axes=(0, 0)))
random_horizontal_flip_jitted = jit(vmap(random_horizontal_flip, in_axes=(0, 0)))
random_vertical_flip_jitted = jit(vmap(random_vertical_flip, in_axes=(0, 0)))


def augment_images(images, key):
    """Augment a batch of input images.

    Args:
        images: Batch of input images as a jax array
        key: Seed/Key for random functions for generating booleans
    Returns:
        Augmented images with the same shape as the input images
    """

    batch_size = len(images)

    # 1. Rotation
    key, subkey = random.split(key)
    rotate = random.randint(key, shape=[batch_size], minval=0, maxval=2)
    augmented = random_rotate_jitted(images, rotate)

    # 2. Flip horizontally
    key, subkey = random.split(key)
    flip = random.randint(key, shape=[batch_size], minval=0, maxval=2)
    augmented = random_horizontal_flip_jitted(augmented, flip)

    # 3. Flip vertically
    key, subkey = random.split(key)
    flip = random.randint(key, shape=[batch_size], minval=0, maxval=2)
    augmented = random_vertical_flip_jitted(augmented, flip)

    return augmented


@jit
def normalise(array):
    return jnp.array(array / 255.)




def data_generator(images, labels, batch_size=128, is_valid=False, key=None):
    """Generates batches of data from a given dataset.

    Args:
        images: Image data represented by a ndarray
        labels: One-hot enocded labels
        batch_size: Number of data points in a single batch
        is_valid: (Boolean) If validation data, then don't shuffle and
                    don't apply any augmentation
        key: PRNG key needed for augmentation
    Yields:
        Batches of images-labels pairs
    """

    # 1. Calculate the total number of batches
    num_batches = int(np.ceil(len(images) / batch_size))

    # 2. Get the indices and shuffle them
    indices = np.arange(len(images))

    if not is_valid:
        if key is None:
            raise ValueError("A PRNG key is required if `aug` is set to True")
        else:
            np.random.shuffle(indices)

    for batch in range(num_batches):
        curr_idx = indices[batch * batch_size: (batch + 1) * batch_size]
        batch_images = images[curr_idx]
        batch_labels = labels[curr_idx]

        if not is_valid:
            batch_images = augment_images(batch_images, key=key)
        yield batch_images, batch_labels


