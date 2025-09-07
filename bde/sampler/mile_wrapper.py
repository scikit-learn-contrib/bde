import jax
import jax.numpy as jnp
import blackjax
from blackjax.mcmc.mclmc import MCLMCInfo

class MileWrapper:
    def __init__(self, logdensity_fn, step_size, L):
        self.algo = blackjax.mclmc(logdensity_fn, step_size=step_size, L=L)

    def init(self, position, rng_key):
        # NOTE: position first, rng_key second
        return self.algo.init(position, rng_key)

    def step(self, rng_key, state):
        return self.algo.step(rng_key, state)

    def sample(self, rng_key, init_position, num_samples, thinning=1, store_states=True):
        keys = jax.random.split(rng_key, num_samples + 1)
        state = self.init(init_position, keys[0]) 

        def one_step(state, key):
            state, info = self.step(key, state)
            return state, (state.position, info)

        state, (positions, infos) = jax.lax.scan(one_step, state, keys[1:])

        if thinning > 1:
            positions = jax.tree_util.tree_map(lambda x: x[::thinning], positions)

        return (positions, infos, state) if store_states else (state.position, infos, state)
