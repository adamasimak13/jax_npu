import os
import ctypes
import numpy as np
import jax
import jax.numpy as jnp
from jax import pure_callback
from functools import wraps

# Set the number of logical devices for JAX to match your NPU shards
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# Load the C++ shared library
try:
    _lib_path = os.path.abspath("libnpu_relu.so")
    _lib = ctypes.CDLL(_lib_path)

    # Define the C++ function signature
    _lib.relu_npu.argtypes = [
        ctypes.POINTER(ctypes.c_float), # float* input
        ctypes.POINTER(ctypes.c_float), # float* output
        ctypes.c_int,                   # int size
        ctypes.c_int                    # int shard_id
    ]
    _lib.relu_npu.restype = None # void
    print(f"[INFO] Successfully loaded '{_lib_path}'")
    print(f"[INFO] JAX is configured with {jax.device_count()} logical devices.")
except (OSError, AttributeError) as e:
    _lib = None
    print(f"[ERROR] Could not load 'libnpu_relu.so'. Have you compiled it? Error: {e}")

# The callback function is defined.

# The HOST-SIDE callback function for a SINGLE SHARD 
def _npu_relu_shard_callback(A_shard, shard_id):
    """
    This function is the bridge to C++ for a SINGLE shard of data.
    """
    if not _lib:
        raise RuntimeError("NPU library not loaded. Cannot proceed.")

    
    # Convert the incoming JAX array to a NumPy array to access .ctypes
    A_shard_np = np.asarray(A_shard)

    # Use the NumPy array for all subsequent operations
    C_shard = np.zeros_like(A_shard_np)
    shard_size = A_shard_np.size

    input_ptr = A_shard_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = C_shard.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    _lib.relu_npu(input_ptr, output_ptr, shard_size, int(shard_id))

    return C_shard

# The custom pmap decorator for the NPU 
def pmap_npu(func_to_parallelize, in_axes=0, axis_name="shard_axis"):
    """
    A custom pmap-like decorator that sends each shard to the NPU.
    """
    @wraps(func_to_parallelize)
    def npu_executor(A_shard):
        shard_id = jax.lax.axis_index(axis_name)

        result_shard = pure_callback(
            _npu_relu_shard_callback,
            jax.ShapeDtypeStruct(A_shard.shape, A_shard.dtype),
            A_shard,
            shard_id
        )
        return result_shard

    return jax.pmap(npu_executor, in_axes=in_axes, axis_name=axis_name)


