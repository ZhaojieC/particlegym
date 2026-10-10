import jax

# Validation tests compare against analytical results at tight tolerances, so
# they run in double precision. Float32 behaviour is tested explicitly.
jax.config.update("jax_enable_x64", True)
