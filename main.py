"""Main unet project file. Run model for training & inference"""
from absl import app, flags, logging
from jax import random, numpy as jnp
from src.model import Unet


def main(argv):
    key1, key2 = random.split(random.PRNGKey(0), 2)
    x = random.uniform(key1, (4, 4))

    model = Unet(features=[3, 4, 5])
    params = model.init(key2, x)
    y = model.apply(params, x)

    print('initialized parameter shapes:\n', jax.tree_map(jnp.shape, unfreeze(params)))
    print('output:\n', y)

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
