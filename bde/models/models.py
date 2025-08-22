import jax
import jax.numpy as jnp

from bde.data.dataloader import DataLoader
from bde.training.trainer import FnnTrainer
from bde.data.dataloader import DataLoader


class Fnn(FnnTrainer):
    """Single FNN that can optionally train itself on init."""

    def __init__(self, sizes, *, x=None, y=None, epochs=0, optimizer=None, init_seed=0, auto_train=False):
        super().__init__()  # init the trainer side (history, etc.)
        self.sizes = sizes
        self.params = None  # will hold initialized weights
        # self.trainer =self._start_training()
        self.init_mlp(seed=init_seed)

        # optional auto-train
        if auto_train and x is not None and y is not None and epochs > 0:
            opt = optimizer or self.default_optimizer()
            # use the inherited fit on THIS model
            FnnTrainer.fit(self, model=self, x=x, y=y, optimizer=opt, epochs=epochs)

    def init_mlp(self, seed):
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, len(self.sizes) - 1)
        params = []
        for k, (m, n) in zip(keys, zip(self.sizes[:-1], self.sizes[1:])):
            W = jax.random.normal(k, (m, n)) / jnp.sqrt(m)
            b = jnp.zeros((n,))
            params.append((W, b))
        self.params = params
        return params

    def _start_training(self):
        trainer = FnnTrainer()
        trainer.default_optimizer()
        data = DataLoader()
        trainer.fit(
            model=self,
            # x=X_true,
            # y=y_true,
            x= data["x_gen"],
            y=data["y_gen"],
            optimizer=trainer.default_optimizer(),  # the default optimizer!
            epochs=1000
        )
        return trainer


