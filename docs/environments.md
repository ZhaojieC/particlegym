# Environments

!!! note "Preview"
    The environment layer is under development for version 0.2
    ([roadmap](roadmap.md)). Observations and rewards will become
    configurable, and Gymnasium and PettingZoo wrappers will follow.

## Room evacuation

`particlegym.envs.Evacuation` simulates pedestrians in a rectangular room with
doors, following the social force model. Some pedestrians are agents: an
action sets an agent's desired direction of motion, and the motion follows
from the physics ([design principle P1](design.md#design-principles)).

```python
import jax
import jax.numpy as jnp

from particlegym import envs
from particlegym import physics as pp

env = envs.Evacuation(
    room=(10.0, 10.0),
    exits=[pp.Opening("y+", (4.5, 5.5))],  # a 1 m door in the top wall
    n_agents=1,
)
params = env.default_params

key = jax.random.key(0)
obs, state = env.reset(key, params)
obs, state, reward, terminated, truncated, info = env.step(key, state, jnp.array([0]), params)
```

The environment is functional: `env` holds the static structure (room, doors,
obstacles, numbers of pedestrians, time steps), `state` and `params` are JAX
pytrees, and `reset` and `step` are pure functions. They can be compiled with
`jax.jit`, batched with `jax.vmap`, and iterated with `jax.lax.scan`:

```python
reset = jax.jit(jax.vmap(env.reset))
step = jax.jit(jax.vmap(env.step))

keys = jax.random.split(key, 1024)  # 1024 environments in parallel
obs, states = reset(keys)
actions = jnp.zeros((1024, 1), dtype=int)
obs, states, reward, terminated, truncated, info = step(keys, states, actions)
```

`step` does not reset finished episodes automatically.

### Room, doors and obstacles

The room spans \([0, \text{width}] \times [0, \text{height}]\). Doors are
`Opening` instances on the faces `"x-"`, `"x+"`, `"y-"` and `"y+"`; the walls
are closed everywhere else, and the door edges act as door jambs. Obstacles
are shape sets, for example `pp.Spheres([[3.0, 3.0]], [1.0])` for a pillar of
radius 1 m.

A pedestrian has left the room once its centre is outside the room by more
than its radius. It is then switched off with the alive mask, and its exit
time is stored in `state.exit_time` (infinite while inside).

### Agents and crowd

The first `n_agents` pedestrians are agents. The other `n_crowd` pedestrians
follow a built-in `crowd_policy`:

| Policy | Desired motion |
|---|---|
| `"nearest_exit"` | towards the nearest door, aiming 1 m beyond its centre |
| `"random"` | a new random direction every control step |
| `"stand"` | none (pedestrians still repel each other) |

All pedestrians walk with the desired speed `params.desired_speed`.

### Actions

| `action` | Action per agent | Desired velocity |
|---|---|---|
| `"discrete"` | an integer in `[0, n_directions)` | \(v_0\) times direction k, at \(90° + 360° k / n\) |
| `"continuous"` | a 2D vector \(\mathbf a\) | \(v_0\, \mathbf a / \max(1, \lVert\mathbf a\rVert)\) |

With the default eight directions, the order is up, up-left, left, down-left,
down, down-right, right and up-right, as in Zhang, Chai & Lykotrafitis (2021);
`env.directions` lists them.

### Observations and rewards

Each agent observes its position, scaled to \([-0.5, 0.5]\) inside the room,
and its velocity: `obs` has shape `(n_agents, 4)`, the state used by Zhang,
Chai & Lykotrafitis (2021).

An agent receives `params.step_penalty` (default -0.1) for every control step
in the room, `params.exit_reward` (default 0) for the step in which it leaves,
and nothing afterwards. `terminated` is true once an agent has left;
`truncated` is true when the episode reaches `params.max_steps` with the agent
still inside.

### Time steps

One control step advances the physics by `substeps` BAOAB steps of length
`dt` (defaults: 20 × 5 ms = 0.1 s). The physics step resolves the stiff
contacts of the social force model; the control interval is the time between
decisions ([design decision D8](design.md#decisions)).

### Parameters

`EvacuationParams` holds the values that may change between episodes:

| Field | Default | Meaning |
|---|---|---|
| `desired_speed` | 2.0 m/s | desired walking speed \(v_0\) |
| `relaxation_time` | 0.5 s | relaxation time \(\tau\) |
| `noise_kT` | 0 J | thermal energy of random velocity fluctuations |
| `step_penalty` | -0.1 | reward per control step inside the room |
| `exit_reward` | 0 | reward for leaving the room |
| `max_steps` | 1000 | episode length in control steps |

Because parameters are a pytree, one compiled function can run a parameter
sweep. For example, eight desired speeds at once:

```python
speeds = jnp.linspace(1.0, 5.0, 8)
params = jax.tree.map(lambda x: jnp.full(8, x), env.default_params)
params = params._replace(desired_speed=speeds)
obs, states = jax.vmap(env.reset)(jax.random.split(key, 8), params)
```
