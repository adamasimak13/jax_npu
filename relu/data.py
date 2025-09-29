import jax
import jax.numpy as jnp
import numpy as np

# Set a fixed seed for reproducibility
key = jax.random.PRNGKey(42)

# Number of elements: 131072 bytes / 4 bytes per float32 = 32768
num_elements = 32768

# Generate random float32 values
data = jax.random.normal(key, (num_elements,), dtype=jnp.float32)

# Convert to NumPy array and save as binary
np_data = np.array(data, dtype=np.float32)
np_data.tofile("input_data.bin")

