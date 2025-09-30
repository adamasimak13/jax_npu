# npu4matmul_multi.py
import os
import ctypes
import numpy as np
import jax
import jax.numpy as jnp
from jax import pure_callback
from concurrent.futures import ThreadPoolExecutor

# Force 3 logical devices
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=3"

print(f"[npu4matmul_multi] Using {jax.device_count()} logical devices")

# --- Load shared library ---
lib = ctypes.CDLL(os.path.abspath("libnpu_matmul.so"))
lib.matmul_npu.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # A
    ctypes.POINTER(ctypes.c_float),  # B
    ctypes.POINTER(ctypes.c_float),  # C
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int                     # shard_id
]
lib.matmul_npu.restype = None


# --- Host callback for multi-kernel execution ---
def matmul_host_parallel(A_shards, B_full):
    num_devices, M_shard, K = A_shards.shape
    _, N = B_full.shape

    C_shards = np.zeros((num_devices, M_shard, N), dtype=np.float32)

    def run_kernel(idx):
        lib.matmul_npu(
            A_shards[idx].astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B_full.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            C_shards[idx].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M_shard, N, K, idx
        )

    # Run 3 kernels in parallel threads
    with ThreadPoolExecutor(max_workers=num_devices) as ex:
        ex.map(run_kernel, range(num_devices))

    return C_shards


# --- JAX wrapper ---
def matmul_npu_parallel(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    It performs matmul on the NPU with data partitioning (pmap) 
    
    """
    num_devices = jax.device_count()
    M_total, K = A.shape
    _, N = B.shape
    M_shard = M_total // num_devices

    # Reshape σε (num_devices, M_shard, K)
    A_shards = A.reshape(num_devices, M_shard, K)

    return pure_callback(
        lambda a, b: matmul_host_parallel(np.array(a), np.array(b)),
        jax.ShapeDtypeStruct((num_devices, M_shard, N), jnp.float32),
        A_shards, B
    )
