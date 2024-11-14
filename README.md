# poisson_sampling

Faster Poisson Sampling in Jax

For lambda = 12.0, shape = (int(2**25),):

- Execution time: 0.006560 seconds
- Jax Baseline Execution time: 0.044150 seconds

The implementation is faster than both original Knuth and rejection sampling for lambda >= 1.02

## Idea
Previous:
```python
def _poisson_rejection(key, lam, shape, dtype, max_iters) -> Array:
  def body_fn(carry):
    ...
    accepted |= accept

    return i + 1, k_out, accepted, key

  def cond_fn(carry):
    i, k_out, accepted, key = carry
    return (~accepted).any() & (i < max_iters)

  k_init = lax.full_like(lam, -1, lam.dtype, shape)
  accepted = lax.full_like(lam, False, jnp.bool_, shape)
  k = lax.while_loop(cond_fn, body_fn, (0, k_init, accepted, key))[1]
  return k.astype(dtype)
```
The original poisson_rejection function uses a while loop to generate poisson samples, which only stops when the last element is sampled, this wasted a lot of samples, i.e., tail/expectation times, which is not efficient especially when the sample size is large. Here we estimate the rejection rate and use the rejection rate to estimate the number of samples needed to generate the last sample, then we generate the samples in one go.