import jax
import jax.numpy as jnp
import blackjax
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from bde.sampler.callbacks import progress_bar_scan

def _infer_dim_from_position_example(pos_e):
    ex = tree_map(lambda a: a[0], pos_e)           # take member 0
    flat, _ = ravel_pytree(ex)
    return flat.shape[0]

def _pad_axis0(a, pad):
    if pad == 0: return a
    return jnp.concatenate([a, jnp.repeat(a[:1], pad, axis=0)], axis=0)

def _reshape_to_devices(a, D, E_per):
    return a.reshape(D, E_per, *a.shape[1:])

class MileWrapper:
    def __init__(self, logdensity_fn):
        self.logdensity_fn = logdensity_fn
        self._kernel_builder = lambda sqrt_diag_cov: blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=self.logdensity_fn,
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
            sqrt_diag_cov=sqrt_diag_cov,
        )
        
    def init(self, position, rng_key):
        return blackjax.mcmc.mclmc.init(position=position,
                                        logdensity_fn=self.logdensity_fn,
                                        rng_key=rng_key)

    def step(self, rng_key, state, L, step_size, sqrt_diag_cov=None):
        if sqrt_diag_cov is None:
            dim = ravel_pytree(state.position)[0].shape[0]
            sqrt_diag_cov = jnp.ones((dim,))
        kernel = self._kernel_builder(sqrt_diag_cov)
        next_state, info = kernel(rng_key=rng_key, state=state, L=L, step_size=step_size)
        return next_state, info

    def init_batched(self, positions_e, keys_e):
        """positions_e: pytree with leading axis E; keys_e: (E,) PRNGKey"""
        init_one = lambda pos, key: blackjax.mcmc.mclmc.init(position=pos,
                                                             logdensity_fn=self.logdensity_fn,
                                                             rng_key=key)
        return jax.vmap(init_one)(positions_e, keys_e)

    def _step_batched(self, keys_e, states_e, L_e, step_e, sqrt_diag_e=None):
        """Vectorized one-step over ensemble axis E."""
        if sqrt_diag_e is None:
            # make ones per member with correct dim
            dim = ravel_pytree(states_e.position)[0].shape[1]  # (E, dim)
            sqrt_diag_e = jnp.ones((states_e.position[0].shape[0], dim))  # fallback; adjust if needed

        def step_one(key, state, L_i, step_i, sdc_i):
            kernel = self._kernel_builder(sdc_i)
            next_state, info = kernel(rng_key=key, state=state, L=L_i, step_size=step_i)
            return next_state, info

        return jax.vmap(step_one, in_axes=(0, 0, 0, 0, 0))(keys_e, states_e, L_e, step_e, sqrt_diag_e)

    def sample_batched(self, rng_keys_e, init_positions_e, num_samples, thinning=1,
                            L_e=None, step_e=None, sqrt_diag_e=None, store_states=True):
        # --- shapes & checks
        E = jax.tree_util.tree_leaves(init_positions_e)[0].shape[0]
        D = jax.local_device_count()
        if L_e is None or step_e is None:
            raise ValueError("Pass per-member L_e and step_e from warmup.")
        if sqrt_diag_e is None:
            dim = _infer_dim_from_position_example(init_positions_e)
            sqrt_diag_e = jnp.ones((E, dim))

        # --- pad to multiple of devices
        pad = (D - (E % max(D, 1))) % max(D, 1)
        E_pad = E + pad
        E_per = E_pad // max(D, 1)

        rng_keys_e = _pad_axis0(rng_keys_e, pad)
        L_e        = _pad_axis0(L_e, pad)
        step_e     = _pad_axis0(step_e, pad)
        sqrt_diag_e= _pad_axis0(sqrt_diag_e, pad)
        init_positions_e = tree_map(lambda a: _pad_axis0(a, pad), init_positions_e)

        # --- shard to (D, E_per, ...)
        if D == 0:
            raise RuntimeError("No devices available.")
        keys_de   = rng_keys_e.reshape(D, E_per, *rng_keys_e.shape[1:])
        L_de      = _reshape_to_devices(L_e, D, E_per)
        step_de   = _reshape_to_devices(step_e, D, E_per)
        sdc_de    = _reshape_to_devices(sqrt_diag_e, D, E_per)
        pos_de    = tree_map(lambda a: _reshape_to_devices(a, D, E_per), init_positions_e)

        # --- per-device function: vmap over local chunk
        def sample_chunk(keys_chunk, init_pos_chunk, L_chunk, step_chunk, sdc_chunk):
            # one member
            def sample_one_member(key, init_pos, L_i, step_i, sdc_i):
                keys = jax.random.split(key, num_samples + 1)
                state = self.init(init_pos, keys[0])
                kernel = self._kernel_builder(sdc_i)   # build once per member

                def body(st, k):
                    st, info = kernel(rng_key=k, state=st, L=L_i, step_size=step_i)
                    return st, (st.position, info)

                st, (pos_T, info_T) = jax.lax.scan(body, state, keys[1:])
                return pos_T, info_T, st

            return jax.vmap(sample_one_member, in_axes=(0, 0, 0, 0, 0), out_axes=(0, 0, 0))(
                keys_chunk, init_pos_chunk, L_chunk, step_chunk, sdc_chunk
            )

        # --- pmap across devices
        positions_dET, infos_dET, states_dE = jax.pmap(
            sample_chunk, in_axes=(0, 0, 0, 0, 0), out_axes=(0, 0, 0)
        )(keys_de, pos_de, L_de, step_de, sdc_de)

        # --- back to (E_pad, T, ...)
        positions_eT = tree_map(lambda a: a.reshape(E_pad, *a.shape[2:]), positions_dET)
        infos_eT     = tree_map(lambda a: a.reshape(E_pad, *a.shape[2:]), infos_dET)
        states_e     = tree_map(lambda a: a.reshape(E_pad, *a.shape[2:]), states_dE)

        # --- drop padding
        if pad:
            positions_eT = tree_map(lambda a: a[:E], positions_eT)
            infos_eT     = tree_map(lambda a: a[:E], infos_eT)
            states_e     = tree_map(lambda a: a[:E], states_e)

        # --- thinning (post-hoc)
        if thinning > 1:
            positions_eT = tree_map(lambda x: x[:, ::thinning, ...], positions_eT)
            infos_eT     = tree_map(lambda x: x[:, ::thinning, ...], infos_eT)

        if store_states:
            return positions_eT, infos_eT, states_e
        else:
            return states_e.position, infos_eT, states_e

