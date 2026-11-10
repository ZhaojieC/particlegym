# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Learning baselines for the environments, in pure JAX.

- :mod:`particlegym.baselines.ppo`: proximal policy optimisation with a
  policy shared by all agents.
- :mod:`particlegym.baselines.q_learning`: tabular Q-learning for
  environments with discrete observations and actions.
"""

from particlegym.baselines import ppo, q_learning

__all__ = ["ppo", "q_learning"]
