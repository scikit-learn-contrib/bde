import jax
import jax.numpy as jnp 
from jax.tree_util import tree_map, tree_structure

class BDEPredictor():
    def __init__(self, model, positions, Xte):
        self.model = model
        self.positions = positions
        self.Xte = Xte
    
    def get_preds(self):
        means, sigmas = [], []
        num_samples = self.positions[0][0].shape[0]

        for i in range(num_samples):
            sample_params = tree_map(lambda a: a[i], self.positions)
            self.model.params = sample_params
            pred = self.model.predict(self.Xte)

            mu = pred[..., 0]
            sigma = jax.nn.softplus(pred[..., 1]) + 1e-6

            means.append(mu)
            sigmas.append(sigma)

        means = jnp.stack(means, axis=0)
        means = jnp.mean(means, axis=0)   

        sigmas = jnp.stack(sigmas, axis=0)
        sigmas = jnp.mean(sigmas, axis=0)

        return means, sigmas



