import jax
import jax.numpy as jnp
from matmul4npu import matmul_npu_parallel

# Example 1536x512 matmul (3 shards of 512 rows)
A = jnp.ones((768, 768), dtype=jnp.float32)
B = jnp.ones((768, 768), dtype=jnp.float32)

C = matmul_npu_parallel(A, B)
print("Result shape:", C.shape)
