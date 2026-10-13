# Environments

!!! note "Preview"
    The environment layer is under development for version 0.2
    ([roadmap](roadmap.md)). Gymnasium and PettingZoo wrappers will follow.

| Environment | What the agents do | Status |
|---|---|---|
| [`Evacuation`](#room-evacuation) | pedestrians leave a 2D or 3D room through doors | tested physics; calibration against measured flows is planned |
| [`CounterFlow`](experimental.md#counter-flow-in-a-corridor) | pedestrians pass each other in a corridor | experimental |
| [`BrickWall`](experimental.md#brick-wall) | a paddle keeps a ball in play that removes bricks; state or pixel observations | experimental |
| [`Billiards`](experimental.md#billiards) | a player pockets balls with a cue ball | experimental |
| [`Herding`](experimental.md#herding) | shepherds drive a flock to a target | experimental |
| [`Microswimmer`](experimental.md#microswimmer) | swimmers steer upwards through vortices | experimental |
| [`CellMigration`](experimental.md#cell-migration) | cells move together to a source of chemoattractant | experimental |

"Experimental" means: the mechanics are tested, but the environment has not
been compared with measurements and reproduces no published result in
numbers. The drone-encirclement environment of the gallery lives in its own
repository, [drone-encircle](https://github.com/ZhaojieC/drone-encircle).

## Interface

All environments have the same functional interface:

```python
env = envs.Herding(n_sheep=40)  # static structure
params = env.default_params  # episode parameters, a pytree
obs, state = env.reset(key, params)
obs, state, reward, terminated, truncated, info = env.step(key, state, action, params)
image = env.render(state)  # RGB picture, needs Pillow
```

- `reset` and `step` are pure functions of their arguments. They compile
  with `jax.jit`, batch with `jax.vmap` (over keys, states, actions and
  parameters) and iterate with `jax.lax.scan`.
- `env.n_agents` is the number of agents. Observations, actions, rewards and
  the two end-of-episode flags have a leading axis of this length, also for
  a single agent.
- `env.observation_space` and `env.action_space` describe the observation and
  the action of *one* agent: `Discrete(n)` or `Continuous(low, high, shape)`,
  each with `sample(key, batch)` and `contains(x)`.
- `step` does not reset finished episodes.
- An action sets what an agent *wants*; the motion follows from forces and an
  integrator ([design principle P1](design.md#design-principles)).

## Room evacuation

`particlegym.envs.Evacuation` simulates pedestrians in a rectangular room with
doors, following the social force model. Some pedestrians are agents: an
action sets an agent's desired direction of motion, and the motion follows
from the physics ([design principle P1](design.md#design-principles)). The
room is two- or three-dimensional, depending on the number of entries of
`room`; the 2D room comes first, and the [3D room](#three-dimensional-rooms)
is described below.

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

The room spans \([0, L_a]\) along every axis, with the edge lengths \(L_a\)
given by `room`; `env.dim` is the number of dimensions d and `env.size` holds
the edge lengths. In 2D, doors are `Opening` instances on the faces `"x-"`,
`"x+"`, `"y-"` and `"y+"`; the walls are closed everywhere else, and the door
edges act as door jambs. Obstacles are shape sets, for example
`pp.Spheres([[3.0, 3.0]], [1.0])` for a pillar of radius 1 m.

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
| `"discrete"` | an integer in `[0, n_directions)` | \(v_0\) times direction k of `env.directions` |
| `"continuous"` | a d-dimensional vector \(\mathbf a\) | \(v_0\, \mathbf a / \max(1, \lVert\mathbf a\rVert)\) |

In 2D, direction k points at \(90° + 360° k / n\). With the default eight
directions, the order is up, up-left, left, down-left, down, down-right, right
and up-right, as in Zhang, Chai & Lykotrafitis (2021). The directions of a 3D
room are listed [below](#three-dimensional-rooms).

### Observations and rewards

Each agent observes its position, scaled to \([-0.5, 0.5]\) inside the room,
and its velocity: `obs` has shape `(n_agents, 2 d)`. In 2D these are the four
numbers of the state used by Zhang, Chai & Lykotrafitis (2021).

An agent receives `params.step_penalty` (default -0.1) for every control step
in the room, `params.exit_reward` (default 0) for the step in which it leaves,
and nothing afterwards. `terminated` is true once an agent has left;
`truncated` is true when the episode reaches `params.max_steps` with the agent
still inside.

### Three-dimensional rooms

A `room` with three edge lengths gives a 3D room. Everything above carries
over, with one more coordinate: positions, velocities and continuous actions
have three components, and `obs` has shape `(n_agents, 6)`.

```python
env = envs.Evacuation(
    room=(10.0, 10.0, 4.0),
    exits=[
        pp.Opening("y+", ((4.5, 5.5), (0.0, 2.0))),  # a door 1 m wide and 2 m high
        pp.Opening("z+", ((4.0, 6.0), (4.0, 6.0))),  # a 2 m x 2 m hatch in the ceiling
    ],
    n_agents=40,
)
```

!!! note "No gravity"
    In 3D the pedestrians are spheres that move freely in all three
    directions. This describes swimmers and other self-propelled particles
    that leave a chamber. It does not describe people walking on a
    floor: for those, use the 2D room. Buildings with several storeys are on
    the [roadmap](roadmap.md).

**Doors.** The faces `"z-"` (floor) and `"z+"` (ceiling) are available in
addition to the four side walls. The span of an `Opening` is one
`(start, end)` pair for each of the two other axes, in increasing axis order:
`(x, y)` for a z face, `(x, z)` for a y face and `(y, z)` for an x face. A
face can have one door.

**Discrete actions.** The directions point from the centre of a cube to its
faces, edges and corners:

| `n_directions` | Directions |
|---|---|
| 6 | the 6 faces: ±x, ±y, ±z |
| 18 | the faces and the 12 edges |
| 26 (default) | the faces, the edges and the 8 corners |

`env.directions` lists them in this order: first the directions in the xy
plane, in the order of the 2D room (eight of them, or four for
`n_directions=6`); then +z, followed by the directions tilted towards +z; then
-z, followed by the directions tilted towards -z. An agent that uses only the
first eight actions of the default set therefore moves as in the 2D room.

**Crowd.** With `"random"`, the directions are uniformly distributed on the
sphere. With `"nearest_exit"`, pedestrians head for a point 1 m beyond the
centre of the nearest door or hatch.

**Consistency with 2D.** Pedestrians in the mid-plane of a 3D room whose doors
span the full height, with the floor and the ceiling beyond the range of the
wall force, follow the same trajectories as in the 2D room
([Validation](validation.md#crowd-dynamics)).

### Time steps

One control step advances the physics by `substeps` BAOAB steps of length
`dt` (defaults: 20 × 5 ms = 0.1 s). The physics step resolves the stiff
contacts of the social force model; the control interval is the time between
decisions ([design decision D8](design.md#decisions)).

**Dense crowds.** The sliding friction is integrated explicitly. For a
pedestrian that overlaps a wall by \(\delta\), every physics step multiplies
its sliding velocity by \(1 - \kappa\,\delta\,\Delta t/m\), so the step is
unstable when \(\kappa\,\delta\,\Delta t/m > 2\). With the defaults
(\(\kappa = 2.4\times10^5\) kg m⁻¹ s⁻¹, \(m = 80\) kg, \(\Delta t = 5\) ms)
this is an overlap of 0.13 m with a wall, and 0.067 m between two pedestrians
of equal mass. Such overlaps can occur where a large crowd presses against a
door post: a run with 10,000 pedestrians and a desired speed of 2 m/s became
unstable after 9.6 s at a door post, while the same run at 1.3 m/s stayed
stable (`examples/gallery/crowd_10000.py`). For crowds of thousands, use a
smaller `dt` with more `substeps` or a lower desired speed, and check the
largest speed of the pedestrians.

### Pictures and animations

`env.render(state)` draws a 2D room and returns the picture as a NumPy array
of shape (height, width, 3). It runs on the host, outside compiled code, and
needs Pillow (`pip install "particlegym[viz]"`). Walls are black, obstacles
grey, agents red and the crowd blue; pass `colors` to choose the colour of
every pedestrian.

```python
from particlegym import viz

frames = [env.render(state)]
for key in jax.random.split(key, 100):
    obs, state, *_ = step(key, state, action)
    frames.append(env.render(state))
viz.save_gif("room.gif", frames, fps=20)  # or viz.save_mp4, which needs ffmpeg
```

`particlegym.viz.Canvas` draws discs, rectangles, lines and the shape sets of
the physics core for any 2D system. For image observations inside compiled
code, `viz.pixel_centres`, `viz.disc_coverage` and `viz.box_coverage` compute
small pixel images in JAX. 3D systems are written with
`particlegym.io.write_extxyz` and rendered with OVITO; the scripts of the
[gallery](gallery.md) show how.

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
