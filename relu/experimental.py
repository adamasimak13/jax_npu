import relu4npu
import jax
import jax.numpy as jnp
import numpy as np

# Define a ReLU function
def relu(x):
  """A standard ReLU function. Our patch will redirect this to the NPU."""
  return jnp.maximum(x, 0)

#  Use the custom pmap for the NPU
parallel_relu = relu4npu.pmap_npu(relu)

# --- Example Execution ---

num_devices = jax.device_count()
print(f"\n--- Running on {num_devices} devices ---")
input_file = "input_data.bin"
data_type = np.float32
try:
    flat_data = np.fromfile(input_file, dtype=data_type)
    print(f"Successfully read {flat_data.size} elements from '{input_file}'.")
except FileNotFoundError:
    print(f"Error: File '{input_file}' not found.")
    print("Please run 'data.py' first to generate it.")
    exit()
if flat_data.size % num_devices != 0:
    print(f"Error: The number of elements ({flat_data.size}) is not evenly divisible by the number of devices ({num_devices}).")
    exit()
x_jax = jnp.array(flat_data).reshape((num_devices, -1))
print("Input shape:", x_jax.shape)
print("Input data sample:", x_jax.flatten()[:5])
y_jax = parallel_relu(x_jax)
y_jax.block_until_ready()
print("\nNPU execution complete.")
print("Output shape:", y_jax.shape)
print("Output data sample:", y_jax.flatten()[:5])