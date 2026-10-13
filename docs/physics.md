# Physics models

This page gives the equations behind every model in `particlegym.physics`,
the meaning and units of the parameters, and the references.

## Equation of motion

Particle \(i\) with mass \(m_i\), position \(\mathbf x_i\) and velocity
\(\mathbf v_i\) obeys the Langevin equation with a drift velocity
\(\mathbf u_i\):

\[
m_i \dot{\mathbf v}_i
  = \mathbf F_i(\mathbf x) + \mathbf F^{\mathrm d}_i(\mathbf x, \mathbf v)
  - m_i \gamma_i (\mathbf v_i - \mathbf u_i)
  + \sqrt{2 m_i \gamma_i k_B T}\, \boldsymbol\xi_i(t),
\]

where \(\mathbf F_i = -\nabla_i U\) is the conservative force,
\(\mathbf F^{\mathrm d}_i\) the dissipative contact force, \(\gamma_i\) a
friction rate, \(k_B T\) the thermal energy and \(\boldsymbol\xi_i\) Gaussian
white noise with unit variance.

| System | \(\gamma\) | \(\mathbf u\) | \(k_B T\) |
|---|---|---|---|
| Pedestrians (social force model) | \(1/\tau\), relaxation time \(\tau\) | \(v_0 \mathbf e_i\), desired speed and direction | 0, or small for fluctuations |
| Passive Brownian particles | friction rate | 0 | thermal energy |
| Self-propelled (active) particles | friction rate | propulsion velocity | thermal energy |
| Hamiltonian dynamics (NVE) | 0 | — | — |

For pedestrians, the friction term is exactly the driving term
\(m_i (v_0 \mathbf e_i - \mathbf v_i)/\tau\) of Helbing et al. (2000). The
desired direction \(\mathbf e_i\) is what an RL agent controls.

## Pair potentials

A pair potential is a function \(\phi(r, \sigma)\) of the centre distance
\(r\) and the contact distance \(\sigma = r_i + r_j\) (the sum of the radii).
The force on particle \(i\) from particle \(j\) is
\(-\phi'(r)\,\hat{\mathbf r}_{ij}\) with \(\hat{\mathbf r}_{ij}\) pointing from
\(j\) to \(i\); \(\phi'\) is computed by automatic differentiation.

**Harmonic** (`Harmonic(k)`), soft repulsion of overlapping particles:

\[
\phi(r) = \tfrac{k}{2} (\sigma - r)^2 \quad (r < \sigma), \qquad 0 \text{ otherwise.}
\]

**WCA** (`WCA(epsilon)`), the repulsive part of the Lennard-Jones potential
(Weeks, Chandler & Andersen, 1971):

\[
\phi(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right] + \varepsilon
\quad (r < 2^{1/6}\sigma), \qquad 0 \text{ otherwise.}
\]

Energy and force vanish continuously at the cut-off.

**Social force** (`SocialForce(A, B, k, interaction_range)`), the conservative
part of the granular social force model (Helbing, Farkas & Vicsek, 2000). The
force magnitude is

\[
f(r) = A\, e^{(\sigma - r)/B} + k\, g(\sigma - r), \qquad g(x) = \max(x, 0),
\]

an exponential "social" repulsion plus a body force during contact. It derives
from

\[
\phi(r) = A B \left[e^{(\sigma - r)/B} - e^{-\ell/B}\right] + \tfrac{k}{2}\, g(\sigma - r)^2,
\]

truncated at \(r = \sigma + \ell\) with \(\ell\) = `interaction_range`. The
constant makes the energy continuous; the neglected force at the cut-off is
\(A e^{-\ell/B}\) (below 0.01 N for the default values).

## Dissipative contact forces

**Tangential friction** (`TangentialFriction(kappa)`), the sliding friction of
Helbing et al. (2000) between touching particles:

\[
\mathbf F^{\mathrm d}_{i \leftarrow j} = \kappa\, g(\sigma - r)\,
\left[\Delta\mathbf v - (\Delta\mathbf v\cdot\hat{\mathbf r}_{ij})\hat{\mathbf r}_{ij}\right],
\qquad \Delta\mathbf v = \mathbf v_j - \mathbf v_i .
\]

It is antisymmetric (momentum is conserved) and its power is
\(-\kappa\, g\, |\Delta\mathbf v_t|^2 \le 0\).

## Walls and obstacles

Every shape provides the distance \(d\) from a particle centre to its surface:
plates (segments in 2D, rectangles in 3D; unsigned distance, so both sides
repel), spheres and boxes (signed distance, negative inside). The wall energy
of particle \(i\) is \(\sum_m \phi(d_{im}, r_i)\) with any pair potential
\(\phi\), so contact begins at \(d = r_i\). The optional wall friction is

\[
\mathbf F^{\mathrm d}_i = -\kappa\, g(r_i - d)\,
\left[\mathbf v_i - (\mathbf v_i\cdot\mathbf n)\mathbf n\right],
\]

with \(\mathbf n = \nabla d\) the surface normal. `box_walls` builds the walls
of a room from plates and leaves real openings (doors), whose edges act as
door jambs.

## More pair terms

**Contact exponential** (`ContactExponential(B, E)`): the force
\(B\,e^{(\sigma - r)/E}\) acts only for \(r < \sigma\). Zhang, Chai, Sun &
Lykotrafitis (2020) use it for the contact inhibition of locomotion between
cells.

**Species** (`Mixture(potential, strength)`): the energy of a pair of species
\(a\) and \(b\) (the `kind` labels of the particles) is
\(s_{ab}\,\phi(r, \sigma)\) with a symmetric matrix \(s\).

**Normal damping** (`NormalDamping(gamma)`) acts on the approach velocity of
touching particles,
\(\mathbf F^{\mathrm d}_{i \leftarrow j} = \gamma\,(\Delta\mathbf v\cdot\hat{\mathbf r}_{ij})\,\hat{\mathbf r}_{ij}\)
for \(r < \sigma\). With the harmonic potential of stiffness \(k\) this is the
linear spring-dashpot contact, which has a constant coefficient of restitution

\[
e = \exp\!\left(-\frac{\pi\zeta}{\sqrt{1 - \zeta^2}}\right), \qquad
\zeta = \frac{\gamma}{2\sqrt{k\,m^*}}, \qquad m^* = \frac{m_i m_j}{m_i + m_j}.
\]

`damping_for_restitution(e, k, m*)` returns the \(\gamma\) for a wanted \(e\).
The same damping is available for walls (`Walls(damping=...)`, with \(m^*\) the
particle mass).

## Moving and removable bodies

Shapes may be built from the arrays of the environment state inside the
compiled step, for example a paddle (`Spheres`) at its current position or a
piston (`Plates`). Build the `Walls` and the `ForceField` again at every step;
this costs nothing at run time. `Walls(active=mask)` switches single shapes
off (bricks that have been removed, doors that have closed), and
`Walls(velocity=...)` makes the friction and damping forces use the velocity
relative to the moving shape. A particle that is hit by a wall moving with
velocity \(V\) leaves with \(v' = 2V - v\) along the normal.

## External forces and flow fields

`ForceField(external=[...])` adds forces on single particles: `Gravity(g)`
with \(\mathbf F = m\mathbf g\), and `KineticFriction(a, v0)`, a friction of
constant strength against the motion, \(\mathbf F = -m a\,\mathbf v /
\sqrt{v^2 + v_0^2}\), for bodies that slide or roll on a table.

A surrounding fluid with a prescribed velocity field \(\mathbf u(\mathbf x)\)
enters through the drift velocity of the BAOAB integrator: the friction term
\(-m\gamma(\mathbf v - \mathbf u)\) is the Stokes drag. Available fields:
`UniformFlow`, `ShearFlow` and `TaylorGreen`, the array of counter-rotating
vortices \(\mathbf u = \tfrac{u_0}{2}(-\cos kx \sin ky,\ \sin kx \cos ky)\).
The particles do not act back on the flow.

## Active particles

In 2D a self-propelled particle has a heading angle \(\theta\), kept by the
environment next to the particle state. `particlegym.physics.active` provides

\[
d\theta = \omega\,dt + \sqrt{2 D_r}\,dW
\]

(`rotate`), the steering rate \(\omega = \beta\sin(\theta^* - \theta)\) towards
a target direction (`steering_rate`), the mean heading of the neighbours
within a radius, which is the alignment rule of Vicsek et al. (1995)
(`mean_heading`), and the polar order parameter (`polar_order`). The
self-propulsion velocity \(v_0(\cos\theta, \sin\theta)\) is passed to the
integrator as the drift velocity. Without interactions the mean-square
displacement is \(2 v_0^2 D_r^{-2}(D_r t - 1 + e^{-D_r t})\).

## Bonded terms

`ForceField(bonded=[...])` adds energies of listed particles:

| Term | Energy |
|---|---|
| `Bonds(pairs, k, r0)` | \(\tfrac{k}{2}(r - r_0)^2\) |
| `Angles(triplets, k, theta0)` | \(k\,[1 - \cos(\theta - \theta_0)]\), about \(\tfrac{k}{2}(\theta - \theta_0)^2\) near the rest angle |
| `AreaConstraint(rings, k, area0)` (2D) | \(\tfrac{k}{2}(A - A_0)^2\) for the polygon through the particles of a ring |

In 2D the angle is signed. A ring of bonds with an area constraint is a simple
vesicle or cell.

## Dissipative particle dynamics

`PairThermostat(gamma, kT, dt)` is the pairwise thermostat of dissipative
particle dynamics (DPD). Every pair closer than \(\sigma\) feels

\[
\mathbf F^{\mathrm D}_{i \leftarrow j} = \gamma\,w^2\,(\Delta\mathbf v\cdot\hat{\mathbf r}_{ij})\,\hat{\mathbf r}_{ij},
\qquad
\mathbf F^{\mathrm R}_{i \leftarrow j} = \sqrt{\frac{2\gamma k_B T}{\Delta t}}\; w\,\theta_{ij}\,\hat{\mathbf r}_{ij},
\qquad w = 1 - \frac{r}{\sigma},
\]

with \(\theta_{ij} = \theta_{ji}\) of zero mean and unit variance. Both forces
act along the line of centres and are antisymmetric, so momentum is conserved
locally and the fluid has hydrodynamic behaviour. With the conservative force
`Harmonic(k=a)` between particles of radius 0.5, \(F = a(1 - r)\), this is the
standard DPD fluid (Groot & Warren, 1997). The symmetric random numbers are
computed from a hash of the two particle indices and the random key of the
step; they are uniformly distributed and identical for the dense and the
cell-list evaluation. Use `velocity_verlet(..., key=key)`, which evaluates
the dissipative force with the half-step velocity.

`pressure(state, force_field)` returns \((2K + W)/(dV)\) with the virial \(W\)
of the conservative pair forces.

## Scalar fields

`Grid(lo, hi, shape)` holds a field with one value per cell, for example the
concentration of a chemical. Particles add to it (`deposit`) and read it and
its gradient at their positions (`sample`, `sample_gradient`); `diffuse`
advances

\[
\partial_t C = D\nabla^2 C - \lambda C + S
\]

by one explicit step (forward in time, central in space), which is stable for
\(\Delta t \le \left[2D\sum_k \Delta x_k^{-2}\right]^{-1}\). Boundaries are
closed (no flux) or periodic. Deposit and sampling use the same linear
weights, so a particle away from the boundaries feels no gradient from its own
deposit.

## Integrators

**Velocity Verlet** (`velocity_verlet`) integrates Newton's equations. With
conservative forces it is symplectic and time-reversible; the energy error is
bounded and scales as \(\Delta t^2\).

**BAOAB Langevin** (`langevin_baoab`) splits one step into a half kick (B), a
half drift (A), the exact solution of the friction, drift and noise part (O),
another half drift and half kick (Leimkuhler & Matthews, 2013). The O step is

\[
\mathbf v \leftarrow \mathbf u + c_1 (\mathbf v - \mathbf u)
  + \sqrt{(1 - c_1^2)\,k_B T/m}\;\boldsymbol\eta, \qquad c_1 = e^{-\gamma\Delta t},
\]

with standard normal \(\boldsymbol\eta\). Because it is exact, the step is
stable for any \(\gamma\Delta t\); for \(\gamma = 0\) BAOAB reduces to velocity
Verlet. Velocity-dependent forces are evaluated with the velocities after the
O step; in velocity Verlet they are evaluated with the velocities at the half
step.

## Units

The code is unit-agnostic; parameters must be given in one consistent system.

| Use case | Length | Time | Mass | Energy |
|---|---|---|---|---|
| Pedestrians | m | s | kg | J |
| Soft matter (reduced Lennard-Jones units) | \(\sigma\) | \(\sigma\sqrt{m/\varepsilon}\) | \(m\) | \(\varepsilon\) |

Default social-force parameters (Helbing et al., 2000), SI units:

| Parameter | Symbol | Default |
|---|---|---|
| Social repulsion strength | \(A\) | 2000 N |
| Social repulsion range | \(B\) | 0.08 m |
| Body-force stiffness | \(k\) | 1.2 × 10⁵ kg s⁻² |
| Sliding friction | \(\kappa\) | 2.4 × 10⁵ kg m⁻¹ s⁻¹ |
| Typical mass, radius | \(m, r\) | 80 kg, 0.25–0.35 m |
| Relaxation time, desired speed | \(\tau, v_0\) | 0.5 s, 0.6–1.5 m s⁻¹ (higher in panic) |

With these stiffnesses, contacts need time steps of a few milliseconds
(\(\Delta t \approx 0.005\) s works well).

## Known limitations (v0.1)

- Orientations exist in 2D only, as a heading angle kept by the environment;
  there are no torques between particles.
- Hydrodynamic interactions exist only in the DPD fluid; prescribed flow
  fields are not changed by the particles.
- Explicit dissipative forces limit the time step: the sliding friction of
  the social force model (see the environments page), contact damping and the
  DPD thermostat.
- Coincident particles (\(r = 0\)) exert no force on each other, because the
  direction is undefined.
- Geometry is not periodic.

## References

- D. Helbing and P. Molnár, Social force model for pedestrian dynamics,
  *Phys. Rev. E* 51, 4282 (1995).
- D. Helbing, I. Farkas and T. Vicsek, Simulating dynamical features of escape
  panic, *Nature* 407, 487–490 (2000).
- J. D. Weeks, D. Chandler and H. C. Andersen, Role of repulsive forces in
  determining the equilibrium structure of simple liquids, *J. Chem. Phys.*
  54, 5237 (1971).
- B. Leimkuhler and C. Matthews, Rational construction of stochastic numerical
  methods for molecular sampling, *Appl. Math. Res. Express* 2013, 34–56
  (2013).
- M. P. Allen and D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
  Oxford University Press (2017).
- Y. Zhang, Z. Chai and G. Lykotrafitis, Deep reinforcement learning with a
  particle dynamics environment applied to emergency evacuation of a room with
  obstacles, *Physica A* 571, 125845 (2021).
- Y. Zhang, Z. Chai, Y. Sun and G. Lykotrafitis, A deep reinforcement learning
  model based on deterministic policy gradient for collective neural crest
  cell migration, arXiv:2007.03190 (2020).
- T. Vicsek, A. Czirók, E. Ben-Jacob, I. Cohen and O. Shochet, Novel type of
  phase transition in a system of self-driven particles, *Phys. Rev. Lett.*
  75, 1226 (1995).
- R. D. Groot and P. B. Warren, Dissipative particle dynamics: Bridging the
  gap between atomistic and mesoscopic simulation, *J. Chem. Phys.* 107, 4423
  (1997).
- S. Colabrese, K. Gustavsson, A. Celani and L. Biferale, Flow navigation by
  smart microswimmers via reinforcement learning, *Phys. Rev. Lett.* 118,
  158004 (2017).
