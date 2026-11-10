# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Animation: a spacecraft flies a Hohmann transfer to an orbit twice as large.

The orbit-transfer environment with its scripted Hohmann controller. The
grey circle is the starting orbit, the green one the target. The spacecraft
leaves a trail; the short orange line is the exhaust while it thrusts.

    python examples/gallery/orbit_transfer.py --output orbit_transfer.gif
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import envs, viz
from particlegym import physics as pp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=float, default=2.0, help="radius of the target orbit")
    parser.add_argument("--steps", type=int, default=100, help="control steps of 0.1 time units")
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--output", default="orbit_transfer.gif")
    args = parser.parse_args()

    env = envs.OrbitTransfer(n_agents=1)
    params = env.default_params._replace(target_radius=args.target)
    _, state = env.reset(jax.random.key(0), params)
    # Start at the bottom of the picture, moving to the right. The forces
    # stored in the state belong to the old position and are recomputed.
    pos = state.particles.pos.at[1].set(jnp.array([0.0, -1.0]))
    vel = state.particles.vel.at[1].set(jnp.array([1.0, 0.0]))
    particles, _ = pp.init_forces(state.particles._replace(pos=pos, vel=vel), env.force_field)
    state = state._replace(particles=particles)
    step = jax.jit(env.step)
    policy = jax.jit(env.hohmann_policy)

    background = (6, 8, 20)
    extent = 1.2 * args.target
    angle = np.linspace(0.0, 2 * np.pi, 241)
    circle = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    trail = [np.asarray(state.particles.pos[1])]
    frames = []
    arrived_at = None
    for i in range(args.steps):
        action = policy(state, params)
        radial, transverse = env._frame(state.particles)
        thrust = np.asarray(action[0, 0] * radial[0] + action[0, 1] * transverse[0])
        _, state, _, _, _, info = step(None, state, action, params)
        if arrived_at is None and bool(info["arrived"][0]):
            arrived_at = i + 1
        here = np.asarray(state.particles.pos[1])
        trail.append(here)

        box = (-extent, -extent), (extent, extent)
        canvas = viz.Canvas(*box, args.width, background=background)
        canvas.polyline(circle, (90, 90, 110), 0.01 * extent)
        canvas.polyline(args.target * circle, (60, 170, 90), 0.01 * extent)
        canvas.discs(np.zeros((1, 2)), env.planet_radius, (70, 120, 220))
        canvas.polyline(np.array(trail), (150, 160, 200), 0.008 * extent)
        if np.linalg.norm(thrust) > 0.0:
            exhaust = here - 0.25 * thrust / max(np.linalg.norm(thrust), 1e-12)
            canvas.segments(here[None], exhaust[None], (255, 150, 50), 0.03 * extent)
        canvas.discs(here[None], 0.035 * extent, (255, 255, 255))
        frames.append(canvas.frame())
    viz.save_gif(args.output, frames, fps=20)
    used = float(state.fuel_used[0])
    print(f"arrived after {arrived_at} steps, delta-v {used:.4f}; {len(frames)} frames")


if __name__ == "__main__":
    main()
