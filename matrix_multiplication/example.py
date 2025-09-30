import jax
import jax.numpy as jnp
from jax import pmap

# Number of devices
num_devices = jax.device_count()

# Create batched input matrices (one per device)
key = jax.random.PRNGKey(0)
A = jax.random.normal(key, (num_devices, 4, 8))  # Shape: [devices, M, K]
B = jax.random.normal(key, (num_devices, 8, 5))  # Shape: [devices, K, N]

# Define a batched matmul function
@pmap
def batched_matmul(a, b):
    return jnp.matmul(a, b)

# Execute on all devices
C = batched_matmul(A, B)

# Result shape: (num_devices, 4, 5)
print(C.shape)
