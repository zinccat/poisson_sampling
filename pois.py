import jax
from jax import lax, numpy as jnp
from jax.random import uniform, split as _split
from functools import partial
from timeit import default_timer as time
from matplotlib import pyplot as plt
import numpy as np

@partial(jax.jit, static_argnums=(2, 3, 4))
def sample_until_enough(key, lam, shape, dtype, max_iters) -> jnp.ndarray:
    N = shape[0]  # Number of required samples

    # Parameters of the rejection algorithm
    log_lam = lax.log(lam)
    b = 0.931 + 2.53 * lax.sqrt(lam)
    a = -0.059 + 0.02483 * b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    v_r = 0.9277 - 3.6224 / (b - 2)

    # Estimate acceptance probability (approximate)
    p_accept = 0.9  # TODO: Dynamically adjust this value as 1 / inv_alpha

    # Total samples to generate (with safety margin)
    total_samples = int(N / p_accept * 1.2)
    total_samples = max(total_samples, N * 2)  # Ensure we have at least 2*N samples

    def rejection_sampler(key, total_samples):
        key, subkey_u, subkey_v = jax.random.split(key, 3)
        u = uniform(subkey_u, (total_samples,), lam.dtype) - 0.5
        v = uniform(subkey_v, (total_samples,), lam.dtype)
        u_shifted = 0.5 - abs(u)

        k = lax.floor((2 * a / u_shifted + b) * u + lam + 0.43)
        s = lax.log(v * inv_alpha / (a / (u_shifted * u_shifted) + b))
        t = -lam + k * log_lam - lax.lgamma(k + 1)

        accept1 = (u_shifted >= 0.07) & (v <= v_r)
        reject = (k < 0) | ((u_shifted < 0.013) & (v > u_shifted))
        accept2 = s <= t
        accept = accept1 | (~reject & accept2)

        return key, k.astype(dtype), accept

    # Generate samples
    key, k_samples, accept_flags = rejection_sampler(key, total_samples)
    accept_idx = jnp.nonzero(accept_flags, size=N, fill_value=int(1e9))[0]
    k_samples = jnp.take(k_samples, accept_idx, fill_value=int(1e9))
    return k_samples

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    lam = 12.0
    shape = (int(2**25),)
    print(shape)
    dtype = jnp.int32
    max_iters = int(1e9)

    sample_until_enough(key, lam, shape, dtype, max_iters)

    # Benchmark the function
    start = time()
    arr = sample_until_enough(key, lam, shape, dtype, max_iters)
    arr.block_until_ready()
    end = time()
    print(f"Execution time: {end - start:.6f} seconds")

    # Plot the histogram
    plt.hist(np.array(arr), bins=100, density=True)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Poisson Distribution")
    plt.savefig("poisson_sampling.png")