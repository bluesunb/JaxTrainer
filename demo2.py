import tensorflow_datasets as tfds
import tensorflow as tf

tf.random.set_seed(42)
tf.config.set_visible_devices([], 'GPU')

train_steps = 1200
eval_every = 200
batch_size = 32

def make_ds():
    train_ds: tf.data.Dataset = tfds.load('mnist', split='train')
    test_df: tf.data.Dataset = tfds.load('mnist', split='test')

    train_ds = train_ds.map(
        lambda sample: {
            "image": tf.cast(sample['image'], tf.float32) / 255,
            "label": sample['label']
        }
    )

    test_ds = test_df.map(
        lambda sample: {
            "image": tf.cast(sample['image'], tf.float32) / 255,
            "label": sample['label']
        }
    )

    train_ds = train_ds.repeat().shuffle(1024)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    return train_ds, test_ds


import jax
import jax.numpy as jp
import flax.linen as nn
from functools import partial

class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))

        x = nn.Conv(64, (3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x
    
def create_model(sample):
    model = Network()
    params = model.init(jax.random.PRNGKey(0), sample['image'])
    return model, params

import optax
from jax_trainer.logger.metrics_pmap import MultiMetric, Accuracy, Average

def create_tx_metric(lr=5e-3, momentum=0.9):
    tx = optax.adamw(lr, b1=momentum)
    metrics = MultiMetric.create(acc=Accuracy(), loss=Average('loss'))
    return tx, metrics

from flax.training.train_state import TrainState
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import replicate, unreplicate

def main():
    def train_step(state: TrainState, batch, metrics: MultiMetric = None):
        def loss_fn(params):
            logits = state.apply_fn(params, batch['image'])
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['label'])
            return loss.mean(), logits
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, 'batch')
        loss = jax.lax.pmean(loss, 'batch')
        metrics = metrics.update(loss=loss, logits=logits, labels=batch['label'])
        return state.apply_gradients(grads=grad), metrics

    def eval_step(state: TrainState, batch, metrics: MultiMetric):
        logits = state.apply_fn(state.params, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['label'])
        loss = jax.lax.pmean(loss, 'batch')
        metrics = metrics.update(loss=loss, logits=logits, labels=batch['label'])
        return metrics

    history = {
        "train_loss": [],
        "train_acc": [],
        "eval_loss": [],
        "eval_acc": []
    }
    train_ds, test_ds = make_ds()
    train_step = jax.pmap(train_step, axis_name='batch')
    eval_step = jax.pmap(eval_step, axis_name='batch')
    
    model, params = create_model(next(train_ds.as_numpy_iterator()))
    tx, metrics = create_tx_metric()
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx)
    state = replicate(state)
    metrics = replicate(metrics)

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        batch = shard(batch)
        state, metrics = train_step(state, batch, metrics)
        for metric, value in metrics.compute().items():
            history[f'train_{metric}'].append(value)
        metrics.reset()

        for test_batch in test_ds.as_numpy_iterator():
            test_batch = shard(test_batch)
            metrics = eval_step(state, test_batch, metrics)

        for metric, value in metrics.compute().items():
            history[f'eval_{metric}'].append(value)
        metrics.reset()

        print(
            f"[Train] step: {step}, "
            f"loss: {history['train_loss'][-1]:.4f}, "
            f"acc: {history['train_acc'][-1] * 100:.4f}, "
        )

        print(
            f"[Eval] step: {step}, "
            f"loss: {history['eval_loss'][-1]:.4f}, "
            f"acc: {history['eval_acc'][-1] * 100:.4f}, "
        )

if __name__ == "__main__":
    main()