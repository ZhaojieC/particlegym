"""Learning baselines for the environments, in pure JAX.

- :mod:`particlegym.baselines.ppo`: proximal policy optimisation with a
  policy shared by all agents.
- :mod:`particlegym.baselines.q_learning`: tabular Q-learning for
  environments with discrete observations and actions.
"""

from particlegym.baselines import ppo, q_learning

__all__ = ["ppo", "q_learning"]
