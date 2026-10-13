# Validation

Every physical claim of ParticleGym is backed by an automated test. The tests
run in double precision on every pull request (`pytest`); statistical tests
are marked `slow` and use fixed random seeds. This page lists what is tested,
the acceptance criterion, and the result of the reference run.

Reference results: version 0.1.0.dev0 (the sections "More physics" and
"Experimental environments": the development version of 0.2), CPU, float64.

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

## More physics (version 0.2)

| Property | Test | Criterion | Reference result |
|---|---|---|---|
| Free fall | `test_external_flow.py::test_free_fall_follows_the_parabola` | \(\mathbf x(t) = \mathbf x_0 + \mathbf v_0 t + \tfrac12\mathbf g t^2\) for two masses; energy conserved | equal to 1e-12 |
| Bouncing ball | `test_bouncing_ball_conserves_energy` (gravity, harmonic floor, 8 s) | energy within 1e-4; the ball returns to its starting height | passes |
| Terminal velocity in a flow | `test_terminal_velocity_in_a_flow` | \(\mathbf v = \mathbf u + \mathbf g/\gamma\) | within 1e-5 |
| Kinetic friction | `test_kinetic_friction_stops_a_sliding_body` | constant deceleration \(a\); stopping distance \(v^2/2a\) | distance within 0.5 % |
| Flow fields | `test_flow_fields_are_consistent` | Taylor-Green flow is divergence-free; vorticity and stream function agree with derivatives of the velocity | equal to 1e-12 |
| Tracer in the vortex flow | `test_tracer_follows_the_streamlines_of_the_vortex_flow` | the stream function stays constant along the path for 40 time units | within 5e-3 (values up to 0.5) |
| Coefficient of restitution | `test_contacts.py::test_two_body_collision_has_the_set_restitution` (masses 1 and 3; \(e\) = 0.9, 0.6, 0.3) | measured \(e\) and energy loss \(\tfrac12 m^* v^2 (1 - e^2)\) within 0.3 %; momentum conserved | passes; the error is of first order in the time step |
| Damped wall | `test_bounce_off_a_damped_wall` | normal velocity times \(-e\), tangential velocity unchanged | within 1e-3 |
| Moving wall and moving disc | `test_bounce_off_a_moving_wall`, `test_bounce_off_a_moving_disc` | \(v' = 2V - v\) along the normal | within 1e-3 |
| Shapes built inside compiled code | `test_shapes_built_from_traced_arrays_match_fixed_shapes` | forces equal those of fixed geometry | equal to 1e-12 |
| Removable shapes, relative wall velocity | `test_inactive_shapes_do_not_act`, `test_wall_friction_uses_the_velocity_relative_to_the_shape` | no force from inactive shapes; no dissipative force on a particle that moves with the wall | exact |
| Active Brownian particle | `test_active.py::test_active_brownian_particle_has_the_analytic_mean_square_displacement` (4000 particles) | mean-square displacement \(2v_0^2D_r^{-2}(D_rt - 1 + e^{-D_rt})\) within 6 % at \(D_r t\) = 0.1, 1, 10; heading correlation \(e^{-D_r t}\) | passes |
| Steering | `test_steering_relaxes_the_heading_to_the_target` | solution of \(\dot\theta = \beta\sin(\theta^* - \theta)\) | within 0.2 % |
| Alignment | `test_alignment_orders_the_headings_at_low_noise_only` (400 particles, density 4) | polar order above 0.9 at weak noise, below 0.15 at full noise | passes |
| Bonded forces | `test_bonded.py::test_forces_are_the_gradient_of_the_energy_and_sum_to_zero` (ring across periodic boundaries) | forces equal minus the gradient; total force zero | equal to 1e-10 |
| Ring with area constraint | `test_ring_conserves_energy_and_relaxes_to_its_rest_shape` | energy conserved within 1e-3; with friction the ring relaxes to the regular polygon with the rest area, the rest bond length and the bending energy \(n k (1 - \cos(2\pi/n))\) | area within 1e-5, bonds within 1e-4, energy within 1e-6 |
| Ideal chain | `test_ideal_chain_has_the_gaussian_end_to_end_distance` (400 chains of 20 beads) | \(\langle R^2\rangle = (n-1)\,d\,k_BT/k\) within 5 %; temperature within 1 % | passes |
| DPD pair noise | `test_dpd.py::test_pair_noise_is_symmetric_with_unit_variance` | symmetric, zero mean, unit variance, no correlation between pairs or steps, same for dense and listed evaluation | passes |
| DPD with a cell list | `test_thermostat_needs_a_key_and_gives_the_same_forces_with_a_cell_list` | forces including the random forces equal brute force; total force zero | equal to 1e-9 |
| DPD momentum and temperature | `test_dpd_fluid_keeps_momentum_and_temperature` (\(\rho = 3\), \(a = 25\), \(\gamma = 4.5\), \(\Delta t = 0.01\), 8 runs) | momentum stays zero; kinetic temperature within 2 % of \(k_BT\) | passes. With 16 runs of 40 time units (`benchmarks/dpd_fluid.txt`) the temperature is 1.001 ± 0.003 at \(\Delta t\) = 0.01, 1.010 ± 0.003 at 0.02 and 1.026 ± 0.003 at 0.04: the explicit thermostat heats the fluid as the time step grows |
| DPD equation of state | `test_excess_pressure_is_nearly_quadratic_in_the_density` (slow) | \((p - \rho k_BT)/(a\rho^2)\) between 0.090 and 0.094 at \(\rho = 3\), between 0.096 and 0.100 at \(\rho = 5\), below the value \(\pi/30\) for uncorrelated particles | see the table below |
| Species | `test_mixture_scales_the_interaction_of_unlike_particles` | unlike pairs scaled by the matrix entry; forces equal minus the gradient; cell list equals brute force | exact |
| Field interpolation and deposit | `test_fields.py::test_linear_field_is_sampled_exactly`, `test_deposit_conserves_the_amount_and_is_the_adjoint_of_sampling` | linear fields and their gradients are exact; deposit conserves the amount | equal to 1e-12 |
| Diffusion | `test_point_source_spreads_with_the_diffusion_law` | amount and centre conserved; variance grows by \(2Dt\) per axis; Gaussian profile | variance within 1e-6; profile within 2 % of its maximum |
| Field boundaries, decay, self-interaction | `test_closed_and_periodic_boundaries_and_decay`, `test_particle_feels_no_gradient_from_its_own_deposit` | closed and periodic boundaries; steady state source/decay; zero gradient from the own deposit | 1e-6; 1e-12 |

Pressure of the DPD fluid (\(a = 25\), \(\gamma = 4.5\), \(k_BT = 1\), cut-off 1,
\(\Delta t = 0.01\), periodic cube of side 4, 4000 to 8000 steps after 1000
discarded; from `benchmarks/dpd_fluid.txt`, the output of
`benchmarks/dpd_fluid.py`):

| \(\rho\) | particles | pressure | \((p - \rho k_BT)/(a\rho^2)\) |
|---|---|---|---|
| 3 | 192 | 23.71 | 0.0918 |
| 3 (side 6) | 648 | 23.65 | 0.0917 |
| 4 | 256 | 42.32 | 0.0958 |
| 5 | 320 | 66.28 | 0.0980 |
| 6 | 384 | 95.51 | 0.0994 |
| 8 | 512 | 169.67 | 0.1010 |

The statistical error of the coefficient is below 1e-4. It rises with the
density towards the value \(\pi/30 = 0.1047\) of uncorrelated particles. A comparison with the coefficient reported by Groot
& Warren (1997) is open: the paper was not available when this was written.

## Experimental environments

The tests of the [experimental environments](experimental.md) check their
mechanics against the equations they are built from, and that a scripted
policy solves each task. They do not compare with measurements.

| Property | Test | Criterion | Reference result |
|---|---|---|---|
| Brownian dynamics | `test_integrators.py::test_brownian_dynamics_diffusion_drift_and_relaxation` | mean displacement \(\mathbf u t\); variance \(2Dt\) with \(D = k_BT/(m\gamma)\) within 7 % (4000 particles); velocity \(\mathbf F/(m\gamma)\) under a constant force; decay of an overlap at the rate \(2k/(m\gamma)\) | passes; force and Euler decay equal to 1e-10 |
| Two DPD fluids separate | `test_dpd.py::test_two_species_with_a_stronger_unlike_repulsion_separate` | fraction of like neighbour pairs stays at 0.5 for equal repulsion and exceeds 0.8 for a = 60 between the species (192 particles, 15 time units) | passes; 0.90 after 12 time units with 1200 particles |
| Lanes in counter-flow | `test_env_counterflow.py::test_lanes_form_and_the_flow_speeds_up` (80 pedestrians, 4 runs of 90 s) | lane order below 0.2 at the start and above 0.8 at the end; the mean speed rises | order 0.08 to 0.98; speed 0.97 m/s after 10 s, 1.30 m/s at the end |
| Rewards and frames of the counter-flow | `test_reward_is_the_progress_along_the_walking_direction`, `test_observations_are_given_in_the_own_frame` | reward equals the displacement along the walking direction; rotating the corridor by 180° and exchanging the groups exchanges the observations | passes |
| Ball of the brick wall | `test_env_brick_wall.py::test_ball_bounces_elastically_between_the_side_walls` | a bounce changes the speed by less than 0.3 % | passes |
| Bricks, lives, pixels | `test_brick_disappears_after_the_contact_and_gives_a_reward`, `test_lost_ball_costs_a_life_and_the_last_one_ends_the_game`, `test_pixel_observation_shows_the_ball_the_bricks_and_the_last_frame` | one reward per brick; game over after three lost balls; images show the state | passes |
| Scripted brick-wall player | `test_scripted_player_clears_the_wall` (8 games of 5000 steps) | no ball lost, at least 35 of 40 bricks | 39 or 40 bricks in every game |
| Restitution of billiard balls | `test_env_billiards.py::test_head_on_collision_has_the_set_restitution` | measured coefficient equals the set one within 0.4 % (0.95) and 2 % (0.7) | passes |
| Elastic collision of equal balls | `test_elastic_balls_separate_at_a_right_angle` | velocities perpendicular after the collision | passes |
| Friction and cushions | `test_friction_and_cushion` | stopping distance \(v^2/(2a)\); cushion returns the set fraction of the speed within 1 % | passes |
| Pockets | `test_pocketing_a_ball_and_the_cue_ball` | reward 1 per ball, penalty and return of the cue ball after a scratch, end of the game | passes |
| Aiming billiards player | `test_aiming_player_clears_most_of_the_table` (16 games of 30 shots) | more than 12 of 15 balls on average | 14.25 |
| Flock rules | `test_env_herding.py::test_sheep_flee_graze_gather_and_keep_apart` | sheep move away from a shepherd they see, graze otherwise, gather, and keep apart | passes |
| Scripted shepherds | `test_scripted_shepherd_brings_the_flock_to_the_target`, `test_two_shepherds_also_solve_the_task` (40 sheep) | all games solved within 800 (900) steps; bodies overlap by less than a fifth of a diameter | 430 to 610 steps; smallest centre distance 0.84 m |
| Alignment of a swimmer | `test_env_microswimmer.py::test_swimmer_in_still_fluid_turns_and_swims_as_the_equations_say` | heading equals the exact solution \(\tan(\tfrac{\theta_k - \theta}{2}) \propto e^{-t/(2B)}\) within 5e-3 rad | 3.1e-3 rad |
| Swimmer in the flow | `test_flow_carries_and_turns_the_swimmer` | a passive particle stays on its streamline and turns at half the vorticity | passes |
| Naive swimmers | `test_naive_swimmers_are_trapped_or_rise_depending_on_the_parameters` | no net climb at \(\Phi = 0.3, \Psi = 1\); climb at the swimming speed at \(\Phi = 3, \Psi = 0.1\) within 3 % | 0.000; 0.989 of the swimming speed |
| A single cell | `test_env_cell_migration.py::test_single_leader_moves_as_the_equation_of_motion_says_and_is_paid_per_step` | velocity \(u(1 - e^{-t/\tau})\); arrival after the distance over the speed plus \(\tau\); reward -0.1 per step | velocity within 1e-5 |
| Chemoattractant field | `test_interface_shapes_and_batching`, `test_cells_attract_through_the_field_and_the_local_reward_follows_the_angle` | amount of secreted substance after one decay time; gradient points to the other cell; reward \(e^{-\beta^2} - 1\) | amount within 0.1 %; reward within 1e-4 |
| Contact between cells | `test_contact_inhibition_and_walls_keep_cells_apart` | cells driven against each other or into a corner stay a diameter apart and inside the region | passes |
| Migrating group of cells | `test_gradient_following_leaders_reach_the_source_and_co_attraction_keeps_them_together` (6 games, 9 leaders) | all leaders arrive; with co-attraction the group is more compact and slower | spread 0.69 against 1.01 at step 100 |

## Learning

The checks of the wrappers and of the learning baselines are listed on the
page [Learning](learning.md#checks).

## Pictures

| Property | Test | Criterion | Reference result |
|---|---|---|---|
| Discs are drawn where the particles are | `test_viz.py::test_disc_is_drawn_at_its_position_with_its_area` | centroid and area of the drawn disc | centroid within 0.17 pixels (0.5 without oversampling), area within 1 % |
| A rendered room shows the state | `test_viz.py::test_evacuation_render_matches_the_state` | the pixel at the centre of every pedestrian has its colour; pedestrians that have left are not drawn | exact |
| Pixel images in JAX | `test_viz.py::test_pixel_coverage_matches_areas_and_positions` | summed coverage equals the area of discs and boxes; centroid equals the position | area within 1 %, centroid within 0.001 length units |
| Scalar fields as background | `test_viz.py::test_scalar_field_is_drawn_in_place` | the colours follow the values of the field at the right places | passes |

## Planned validation

- Speed-density relation (fundamental diagram) and bottleneck flow against
  experimental data, with calibrated parameters.
- The "faster-is-slower" effect of Helbing et al. (2000).
- Reproduction of the evacuation results of Zhang, Chai & Lykotrafitis (2021).
- Velocity profile and viscosity of the DPD fluid in a channel flow.
- The learning gain of smart microswimmers of Colabrese et al. (2017).
- Collective migration with follower cells as in Zhang et al. (2020).
- The DPD equation of state against the published coefficient.

## Running the tests

```bash
pip install -e ".[test]"
pytest              # all tests, about 12 minutes on 12 CPU cores
pytest -m "not slow"  # skip the long statistical tests
```
