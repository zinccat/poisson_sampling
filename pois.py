import jax
from jax import lax, numpy as jnp
from jax.random import uniform, poisson
from functools import partial
from timeit import default_timer as time
from matplotlib import pyplot as plt
import numpy as np
from jax import jit

Array = jnp.ndarray

@jit
def num_samples(lam, N, safety_margin=1.2):
    b = 0.931 + 2.53 * lax.sqrt(lam)
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)

    # Estimate acceptance probability (approximate)
    p_accept = 1 / inv_alpha

    # Total samples to generate (with safety margin)
    total_samples = N / p_accept * safety_margin
    return total_samples


@partial(jit, static_argnums=(2, 3, 4, 5))
def _poisson_rejection(key, lam, shape, dtype, max_iters, n) -> Array:
    N = shape[0]  # Number of required samples

    # Parameters of the rejection algorithm
    log_lam = lax.log(lam)
    b = 0.931 + 2.53 * lax.sqrt(lam)
    a = -0.059 + 0.02483 * b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    v_r = 0.9277 - 3.6224 / (b - 2)

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
    key, k_samples, accept_flags = rejection_sampler(key, n)
    accept_idx = jnp.nonzero(accept_flags, size=N, fill_value=int(1e9))[0]
    k_samples = jnp.take(k_samples, accept_idx, fill_value=int(1e9))
    return k_samples

def poisson_new(key, lam, shape, dtype=jnp.int32):
    assert lam >= 1.0
    n = int(num_samples(lam, shape[0]))
    return _poisson_rejection(key, lam, shape, dtype, int(1e9), n)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    lam = 12.0
    shape = (int(2**25),)
    n = int(num_samples(lam, shape[0]))
    print("oversampling factor", n / shape[0])  # oversampling factor
    print(shape)
    dtype = jnp.int32
    max_iters = int(1e9)

    poisson_new(key, lam, shape)
    poisson(key, lam, shape)

    # Benchmark the function
    start = time()
    new_arr = poisson_new(key, lam, shape)
    new_arr.block_until_ready()
    end = time()
    print(f"Execution time: {end - start:.6f} seconds")

    start = time()
    ori_arr = poisson(key, lam, shape)
    ori_arr.block_until_ready()
    end = time()
    print(f"Original Execution time: {end - start:.6f} seconds")

    # Plot the histogram
    all_samples = [ori_arr, new_arr]
    all_samples = np.array(all_samples)
    plt.hist(
        all_samples.T,
        bins=50,
        alpha=0.5,
        label=["Original", "Optimized", "NumPy"],
        density=True,
    )
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Poisson Distribution")
    plt.savefig("poisson_sampling.png")
