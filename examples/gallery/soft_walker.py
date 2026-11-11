# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Animation: a soft robot of voxels walks with a wave of contractions.

The default robot of the soft-walker environment: a trunk of horizontal
actuators (orange), a soft layer (grey) and three legs of vertical actuators
(blue). Its commands follow the open-loop gait of the environment, a sine
wave that runs forward along the body; a voxel is lighter while it lengthens
and darker while it shortens. Only the friction of the ground moves the
robot forward. The view follows the robot; the marks on the ground are one
voxel apart.

    python examples/gallery/soft_walker.py --output soft_walker.gif
"""

import argparse

import jax

from particlegym import envs, viz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--time", type=float, default=30.0, help="simulated time")
    parser.add_argument("--frame-every", type=int, default=2, help="control steps per frame")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--output", default="soft_walker.gif")
    args = parser.parse_args()

    env = envs.SoftWalker()
    params = env.default_params
    _, state = env.reset(jax.random.key(0), params)
    step = jax.jit(env.step)
    policy = jax.jit(env.gait_policy)
    steps = round(args.time / env.control_interval)
    frames = [env.render(state, args.width)]
    for i in range(steps):
        state = step(None, state, policy(state, params), params)[1]
        if (i + 1) % args.frame_every == 0:
            frames.append(env.render(state, args.width))
    viz.save_gif(args.output, frames, fps=20)
    centre, _ = env.centre_of_mass(state.particles)
    print(f"{float(centre[0]):.2f} voxels in {args.time:g} time units; {len(frames)} frames")


if __name__ == "__main__":
    main()
