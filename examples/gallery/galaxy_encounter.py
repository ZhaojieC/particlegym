# ParticleGym, https://github.com/ZhaojieC/particlegym (MIT license).
# Please cite it in published work: particlegym.cite() gives the references.
"""Animation: two disc galaxies pass each other and pull out tidal tails.

The restricted three-body model of Toomre & Toomre (1972): each galaxy is a
point mass at the centre of a disc of stars, and the stars are test
particles that feel the two centres but not each other. The two centres
(equal masses, G M = 1 each) approach on a parabolic orbit with a closest
distance of 1. Both discs turn in the sense of the orbit; the second one is
tilted by 45 degrees. The picture is the projection on the plane of the
orbit, around the centre of mass. At the end the script prints the closest
distance of the two centres and the fate of the stars: bound to their own
centre, captured by the other galaxy, or bound to neither (a star counts as
bound to a centre when its energy relative to it alone is negative).

    python examples/gallery/galaxy_encounter.py --output galaxy_encounter.gif

A. Toomre and J. Toomre, Galactic bridges and tails, Astrophys. J. 178,
623-666 (1972).
"""

import argparse
import math

import jax
import jax.numpy as jnp
import numpy as np

from particlegym import physics as pp
from particlegym import viz

SOFTENING = 0.05


def disc(n_rings: int, inner: float, outer: float, per_unit_radius: float, rng):
    """Positions (n, 3) and velocities on circular orbits in the softened field of G M = 1."""
    radii = np.linspace(inner, outer, n_rings)
    pos, vel = [], []
    for r in radii:
        count = max(6, round(per_unit_radius * r))
        phase = rng.uniform(0, 2 * math.pi)
        angle = phase + 2 * math.pi * np.arange(count) / count
        speed = math.sqrt(r**2 / (r**2 + SOFTENING**2) ** 1.5)
        pos.append(np.stack([r * np.cos(angle), r * np.sin(angle), np.zeros(count)], axis=1))
        vel.append(np.stack([-speed * np.sin(angle), speed * np.cos(angle), np.zeros(count)], 1))
    return np.concatenate(pos), np.concatenate(vel)


def tilt(vectors: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate vectors (n, 3) about the x axis."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return vectors @ rotation.T


def initial_conditions(n_rings: int, per_unit_radius: float, seed: int):
    """The two centres, then the stars of the first and of the second galaxy."""
    rng = np.random.default_rng(seed)
    # Relative orbit: a parabola with periapsis q = 1 for G (M1 + M2) = 2,
    # started at the true anomaly of -120 degrees, 3.46 time units before the
    # closest approach.
    f = math.radians(-120.0)
    r = 2.0 / (1.0 + math.cos(f))
    radial = np.array([math.cos(f), math.sin(f), 0.0])
    transverse = np.array([-math.sin(f), math.cos(f), 0.0])
    relative_pos = r * radial
    relative_vel = math.sin(f) * radial + (1.0 + math.cos(f)) * transverse
    centres = np.stack([-0.5 * relative_pos, 0.5 * relative_pos])
    centre_vel = np.stack([-0.5 * relative_vel, 0.5 * relative_vel])

    pos1, vel1 = disc(n_rings, 0.15, 0.6, per_unit_radius, rng)
    pos2, vel2 = disc(n_rings, 0.15, 0.6, per_unit_radius, rng)
    pos2, vel2 = tilt(pos2, 45.0), tilt(vel2, 45.0)
    pos = np.concatenate([centres, centres[0] + pos1, centres[1] + pos2])
    vel = np.concatenate([centre_vel, centre_vel[0] + vel1, centre_vel[1] + vel2])
    galaxy = np.concatenate([[0, 1], np.zeros(len(pos1), int), np.ones(len(pos2), int)])
    return pos, vel, galaxy


def fates(pos: np.ndarray, vel: np.ndarray, galaxy: np.ndarray) -> np.ndarray:
    """Fractions of the stars of each galaxy (rows) bound to centre 1, to centre 2, to neither."""
    energy = np.stack(
        [
            0.5 * np.sum((vel[2:] - vel[k]) ** 2, axis=1)
            - 1.0 / np.sqrt(np.sum((pos[2:] - pos[k]) ** 2, axis=1) + SOFTENING**2)
            for k in (0, 1)
        ],
        axis=1,
    )
    owner = np.where(energy.min(axis=1) < 0.0, energy.argmin(axis=1), 2)
    stars = galaxy[2:]
    return np.array([[np.mean(owner[stars == g] == k) for k in range(3)] for g in (0, 1)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rings", type=int, default=60)
    parser.add_argument(
        "--density", type=float, default=110.0, help="stars per ring per unit radius"
    )
    parser.add_argument("--time", type=float, default=8.5, help="simulated time")
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--frame-every", type=float, default=0.1, help="simulated time per frame")
    parser.add_argument("--width", type=int, default=420)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="galaxy_encounter.gif")
    args = parser.parse_args()

    pos, vel, galaxy = initial_conditions(args.rings, args.density, args.seed)
    field = pp.ForceField(
        pp.Box((-1e3,) * 3, (1e3,) * 3),
        external=[pp.NewtonGravity(softening=SOFTENING, sources=2)],
    )
    state = pp.create_state(jnp.asarray(pos), jnp.asarray(vel), mass=1.0, radius=0.0)
    state, _ = pp.init_forces(state, field)
    per_frame = round(args.frame_every / args.dt)
    n_frames = round(args.time / args.frame_every)

    @jax.jit
    def advance(s):
        def body(s, _):
            s = pp.velocity_verlet(s, field, args.dt)[0]
            return s, jnp.linalg.norm(s.pos[1] - s.pos[0])

        s, separation = jax.lax.scan(body, s, None, length=per_frame)
        return s, jnp.min(separation)

    background = (6, 8, 20)
    colors = np.array([(120, 175, 255), (255, 175, 95)])[galaxy]
    colors[:2] = (255, 255, 255)
    sizes = np.where(np.arange(len(pos)) < 2, 0.06, 0.013)
    extent = 3.6
    frames = []
    closest = np.inf
    for i in range(n_frames + 1):
        xy = np.asarray(state.pos)[:, :2]  # projection on the plane of the orbit
        box = (-extent, -extent), (extent, extent)
        canvas = viz.Canvas(*box, args.width, background=background)
        canvas.discs(xy, sizes, colors)
        frames.append(canvas.frame())
        if i < n_frames:
            state, separation = advance(state)
            closest = min(closest, float(separation))
    viz.save_gif(args.output, frames, fps=15)
    print(f"{len(pos) - 2} stars, {len(frames)} frames written to {args.output}")
    print(f"closest distance of the two centres: {closest:.4f}")
    final = fates(np.asarray(state.pos), np.asarray(state.vel), galaxy)
    print(f"after t = {n_frames * per_frame * args.dt:g}, fractions bound to")
    print("              centre 1  centre 2  neither")
    for g, name in enumerate(("galaxy 1 (flat)", "galaxy 2 (tilted)")):
        print(f"{name:18s}" + "".join(f"{x:9.3f}" for x in final[g]))


if __name__ == "__main__":
    main()
