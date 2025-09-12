import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

import optax
from bde.loss.loss import GaussianNLL

from typing import (
    Any,
    Optional,
    Sequence,
    Tuple,
)


# TODO: major restructure is needed!
class FnnTrainer:

    def __init__(self):
        """
        #TODO: documentation

        """
        self.history = {}
        self.log_every = 100
        self.keep_best = False
        self.default_optimizer = self.default_optimizer

    def _reset_history(self):
        """
        #TODO: documentation

        Returns
        -------

        """
        self.history = {"train_loss": []}

    def train(
            self,
            model,
            x,
            y,
            optimizer: Optional[optax.GradientTransformation] = None,
            epochs: int = 100,
            loss=None,
            ):
        """
        Generic training loop.
        - model.forward(params, x) must exist
        - loss must implement Loss API (apply_reduced)
        """

        if loss is None:
            loss = self.default_loss()

        self._reset_history()

        params = model.params
        opt_state = optimizer.init(params)

        def loss_fn(p, x, y):
            return loss(p, model, x, y)
          
        @jax.jit
        def step(params, opt_state, x, y):
            loss_val, grads = jax.value_and_grad(loss_fn)(params, x, y)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        for epoch in range(epochs):
            params, opt_state, loss_val = step(params, opt_state, x, y)
            self.history["train_loss"].append(float(loss_val))
            if epoch % self.log_every == 0:
                print(epoch, float(loss_val))

        model.params = params
        return model

    @staticmethod
    def default_optimizer():
        return optax.adam(learning_rate=0.01)

    @staticmethod
    def default_loss():
        return GaussianNLL()
    
