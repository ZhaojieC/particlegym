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
O step.

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

- No rotational degrees of freedom: orientations of active particles and
  rotational diffusion are planned.
- No hydrodynamic interactions (DPD is under consideration).
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
