# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""ParticleGym: fast, physically validated particle-dynamics environments for RL."""

from particlegym import envs, physics
from particlegym.citation import cite

__version__ = "0.2.0"

__all__ = ["__version__", "cite", "envs", "physics"]
