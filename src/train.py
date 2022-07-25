"""Training functions & classes file"""
import jax.numpy as jnp
from flax import linen as nn
import jax
from src.model import BasicUnet
import numpy as np

@jax.jit
def train_step(state, imgs, gt_labels):
    def loss_fn(params):
        logits = BasicUnet().apply({'params': params}, imgs)
        one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes=10)
        loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
        return loss, logits

    (_, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)  # this is the whole update now! concise!
    metrics = compute_metrics(logits=logits,
                              gt_labels=gt_labels)  # duplicating loss calculation but it's a bit cleaner
    return state, metrics


@jax.jit
def eval_step(state, imgs, gt_labels):
    logits = BasicUnet().apply({'params': state.params}, imgs)
    return compute_metrics(logits=logits, gt_labels=gt_labels)


def train_one_epoch(state, dataloader, epoch):
    """Train for 1 epoch on the training set."""
    batch_metrics = []
    for cnt, (imgs, labels) in enumerate(dataloader):
        state, metrics = train_step(state, imgs, labels)
        batch_metrics.append(metrics)

    # Aggregate the metrics
    batch_metrics_np = jax.device_get(
        batch_metrics)  # pull from the accelerator onto host (CPU)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    return state, epoch_metrics_np


def evaluate_model(state, test_imgs, test_lbls):
    """Evaluate on the validation set."""
    metrics = eval_step(state, test_imgs, test_lbls)
    metrics = jax.device_get(metrics)  # pull from the accelerator onto host (CPU)
    metrics = jax.tree_map(lambda x: x.item(), metrics)  # np.ndarray -> scalar
    return metrics


def create_train_state(key, learning_rate, momentum):
    cnn = CNN()
    params = cnn.init(key, jnp.ones([1, *mnist_img_size]))['params']
    sgd_opt = optax.sgd(learning_rate, momentum)
    # TrainState is a simple built-in wrapper class that makes things a bit cleaner
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=sgd_opt)


def compute_metrics(*, logits, gt_labels):
    one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes=10)

    loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == gt_labels)

    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


class Train(nn.module):
    seed: int = 0
    learning_rate: float = 0.1
    momentum: float = 0.9
    num_epochs: int = 2
    batch_size: int = 32

    @nn.compact
    def __call__(self, epochs):
        train_state = create_train_state(jax.random.PRNGKey(self.seed), self.learning_rate,
                                         self.momentum)

        for epoch in range(1, self.num_epochs + 1):
            train_state, train_metrics = train_one_epoch(train_state, train_loader,
                                                         epoch)
            print(
                f"Train epoch: {epoch}, loss: {train_metrics['loss']}, accuracy: {train_metrics['accuracy'] * 100}")

            test_metrics = evaluate_model(train_state, test_images, test_lbls)
            print(
                f"Test epoch: {epoch}, loss: {test_metrics['loss']}, accuracy: {test_metrics['accuracy'] * 100}")
