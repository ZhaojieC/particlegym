# Validation

Every physical claim of ParticleGym is backed by an automated test. The tests
run in double precision on every pull request (`pytest`); statistical tests
are marked `slow` and use fixed random seeds. This page lists what is tested,
the acceptance criterion, and the result of the reference run.

Reference results: version 0.1.0.dev0, CPU, float64.

## Physics core

| Property | Test | Criterion | Reference result |
|---|---|---|---|
| Neighbour search is complete | `test_neighbors.py::test_cell_list_forces_match_brute_force` (2D/3D, periodic/open, dead particles, particles outside the box) | forces and energies of cell list equal brute force | equal to 1e-12 |
| Candidate lists | `test_cell_list_candidates_cover_all_close_pairs_once` | every pair closer than the cut-off is a candidate, no duplicates, no dead particles | passes |
| Forces are minus the energy gradient | `test_potentials.py::test_forces_equal_minus_energy_gradient` (all potentials, walls, obstacles) | agreement with `jax.grad` of the total energy | relative difference 6e-17 |
| Energy function itself | `test_energy_gradient_matches_finite_differences` | central finite differences | relative 1e-5 |
| Published social-force law | `test_social_force_matches_published_law` | \(A e^{(\sigma-r)/B} + k\,g(\sigma-r)\) of Helbing et al. (2000), with and without contact | equal to 1e-12 |
| WCA potential | `test_wca_energy_and_force_vanish_at_cutoff` | \(\phi(\sigma) = \varepsilon\); energy and force vanish at \(2^{1/6}\sigma\) | passes |
| Newton's third law | `test_pair_forces_obey_newtons_third_law` | total pair force vanishes | to round-off |
| Friction is dissipative | `test_tangential_friction_conserves_momentum_and_only_dissipates` | equal and opposite, tangential, power \(= -\kappa g\lvert\Delta v_t\rvert^2\) | equal to 1e-12 |
| Energy conservation (NVE) | `test_integrators.py::test_velocity_verlet_conserves_energy_and_momentum` (WCA fluid, 125 particles, \(\Delta t = 0.004\)) | fluctuation < 2e-3 and drift < 2e-4 of the kinetic energy | fluctuation 4.4e-4, drift 1.2e-5 |
| Momentum conservation | same test | total momentum stays zero | 7e-14 |
| Second-order accuracy | `test_velocity_verlet_energy_error_scales_with_dt_squared` | halving \(\Delta t\) divides energy fluctuations by about 4 | ratio 3.77 |
| BAOAB limit | `test_baoab_without_friction_is_velocity_verlet` | identical step for \(\gamma = 0\) | equal to 1e-12 |
| Thermostat, ideal gas | `test_langevin_thermostat_samples_maxwell_boltzmann` | \(\langle m v^2\rangle = k_BT\) within 2 %, \(\langle v^4\rangle = 3(k_BT/m)^2\) within 5 % | +0.6 %, +1.3 % |
| Thermostat, interacting fluid | `test_langevin_thermostat_with_interactions` | mean temperature within 4 standard errors | 1.292 ± 0.014 (target 1.3) |
| Einstein diffusion | `test_free_diffusion_matches_exact_ornstein_uhlenbeck_msd` | MSD equals \(2dD[t - (1-e^{-\gamma t})/\gamma]\), \(D = k_BT/(m\gamma)\), within 5 % from ballistic to diffusive times | +1.0 %, +1.6 %, +0.4 %, +0.4 % at \(\gamma t\) = 0.2, 1, 4, 20 |
| Exact relaxation to the desired velocity | `test_drift_relaxation_is_exact_without_noise` | \(\mathbf v(t) = \mathbf u + (\mathbf v_0 - \mathbf u)e^{-\gamma t}\) | equal to 1e-12 |
| Exits | `test_dead_particles_are_frozen_and_do_not_interact` | dead particles do not move or interact | passes |
| Float32 | `test_float32_states_stay_float32` | single precision preserved, results finite | passes |
| Large radii in float32 | `test_potentials.py::test_large_radii_in_single_precision_give_finite_forces` | finite forces, gradients and energy with radii of 1000 length units | passes |

## Geometry

| Property | Test | Criterion | Reference result |
|---|---|---|---|
| Distances to segments, rectangles, spheres, boxes | `test_geometry.py` | analytical values | exact |
| Rooms with doors | `test_box_walls_leave_a_door_open_in_2d` / `_3d` | door centre is half a door width from the jambs | exact |
| Elastic wall collision | `test_particle_bounces_off_a_wall_elastically` | no penetration, outgoing speed equals incoming speed, energy conserved | within 5e-4 |
| Doors are open, walls are closed | `test_particles_pass_through_the_door_but_not_the_wall` | a particle aimed at the door leaves; one aimed at the wall comes back | passes |

## Crowd dynamics

| Property | Test | Criterion | Reference result |
|---|---|---|---|
| Evacuation through a door only | `test_social_force.py::test_pedestrians_leave_only_through_the_door` (30 pedestrians, 8 m room, 1.2 m door, \(v_0 = 1.2\) m/s, Helbing parameters) | everybody leaves within 60 s; every pedestrian crosses the wall plane inside the door; nobody passes a wall | evacuation time 33 s; crossings between y = 3.79 and 4.19 m (door 3.4–4.6 m) |
| A 3D room reduces to the 2D room | `test_evacuation_env_3d.py::test_mid_plane_of_a_3d_room_reproduces_the_2d_room` (14 pedestrians in the mid-plane of a 10 m × 10 m × 3 m room, two doors over the full height, 8 s) | positions equal those in the 2D room within 1e-8 m; the height stays constant within 1e-12 m; same exit times and rewards | equal to round-off (difference 0 in double precision) |
| Exits in 3D | `test_evacuation_env_3d.py::test_crowd_leaves_only_through_the_exits` (40 pedestrians, 10 m × 10 m × 4 m room, a door in a side wall and a hatch in the ceiling) | everybody leaves within 60 s, each through the door or the hatch | all 40 leave in 7.4 s: 10 through the door, 30 through the hatch |
| Stability limit of the explicit sliding friction | `test_social_force.py::test_explicit_wall_friction_has_a_stability_limit` (a pedestrian sliding along a wall at a fixed overlap) | every step multiplies the sliding velocity by \(1 - \kappa\,\delta\,\Delta t/m\) | exact (1e-9); with the defaults the factor is below −1 for overlaps above 0.133 m |

The mean flow in this test, 0.76 persons per metre per second, is lower than
the specific flow of 1.3 to 1.9 persons/(m s) measured in bottleneck
experiments. With only 30 pedestrians, the time to walk to the door is a large
part of the evacuation, and every pedestrian heads straight at the door
centre. With 100 pedestrians in a 10 m room and \(v_0 = 1.5\) m/s
(`examples/room_evacuation_2d.py`), the mean flow is 1.68 persons/(m s),
inside the experimental range. These are sanity checks, not a calibration.

## Pictures

| Property | Test | Criterion | Reference result |
|---|---|---|---|
| Discs are drawn where the particles are | `test_viz.py::test_disc_is_drawn_at_its_position_with_its_area` | centroid and area of the drawn disc | centroid within 0.17 pixels (0.5 without oversampling), area within 1 % |
| A rendered room shows the state | `test_viz.py::test_evacuation_render_matches_the_state` | the pixel at the centre of every pedestrian has its colour; pedestrians that have left are not drawn | exact |
| Pixel images in JAX | `test_viz.py::test_pixel_coverage_matches_areas_and_positions` | summed coverage equals the area of discs and boxes; centroid equals the position | area within 1 %, centroid within 0.001 length units |

## Planned validation

- Speed-density relation (fundamental diagram) and bottleneck flow against
  experimental data, with calibrated parameters.
- The "faster-is-slower" effect of Helbing et al. (2000).
- Reproduction of the evacuation results of Zhang, Chai & Lykotrafitis (2021).
- Rotational diffusion of active Brownian particles, once orientations are
  added.

## Running the tests

```bash
pip install -e ".[test]"
pytest              # all tests, about 1.5 minutes on a laptop CPU
pytest -m "not slow"  # skip the long statistical tests
```
