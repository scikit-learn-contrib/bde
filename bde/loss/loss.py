from abc import ABC, abstractmethod
import optax
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Optional


class GaussianNLL():
    def __init__(self, min_sigma=1e-6,
                 map_fn=jax.nn.softplus):  # TODO: [@question] why do we use softplus for the sigma??
        self.min_sigma = min_sigma
        self.map_fn = map_fn

    def __call__(self, preds, y_true):
        mu = preds[..., 0:1]
        sigma = self.map_fn(preds[..., 1:2]) + self.min_sigma
        resid = (y_true - mu) / sigma
        nll = 0.5 * (jnp.log(2 * jnp.pi) + 2 * jnp.log(sigma) + resid ** 2)
        return jnp.mean(nll)
