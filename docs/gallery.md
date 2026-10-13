# Gallery

Every animation is a simulation with ParticleGym, made by a script in
[`examples/gallery`](https://github.com/ZhaojieC/particlegym/tree/main/examples/gallery)
with a fixed random seed.

## Crowds

![Pedestrians leave a room through two doors](assets/gallery/evacuation_room.gif)

100 pedestrians leave a room of 14 m × 10 m through two doors of 1.2 m; a
pillar stands in front of the right door. Every pedestrian is an agent that
walks to the nearest door. Colours show the door used. The animation runs at
four times real speed.

![10,000 pedestrians leave a hall](assets/gallery/crowd_10000.gif)

10,000 pedestrians leave a hall of 120 m × 90 m through ten doors of 6 m, at a
desired speed of 1.3 m/s. Colours show the nearest door at the start. The
animation runs at six times real speed. The pedestrians pass the doors faster
than real crowds do: the parameters of the social force model are the
published ones and have not been calibrated against measured flows yet (see
the [roadmap](roadmap.md)).

The two groups of pedestrians below walk in opposite directions along a
corridor that is periodic along its length. Every pedestrian wants to walk
straight on; the lanes form through the interactions.

![Two groups of pedestrians form lanes](assets/gallery/counterflow.gif)

## Games

Arcade and table games built from the same physics core
([details](experimental.md)): the ball and the balls move by Newton's
equations, and every bounce is a contact force. The players in these
animations follow simple scripted rules.

![A scripted player knocks bricks out of a wall](assets/gallery/brick_wall.gif)

![A simple player clears a billiards table](assets/gallery/billiards.gif)

## Animals, swimmers and cells

![A scripted shepherd drives a flock to the target](assets/gallery/herding.gif)

A shepherd (red) collects a flock of 40 sheep and drives it to the target
(green). Flock rules adapted from Strömbom et al. (2014).

![Swimmers that steer upwards are trapped by vortices](assets/gallery/microswimmer.gif)

400 swimmers that always steer upwards in a field of vortices (grey: the
vorticity), the task of Colabrese et al. (2017). The vortices trap them; a
swimmer that senses the vorticity can learn to escape.

![Cells migrate together to a source](assets/gallery/cell_migration.gif)

Nine cells move to a source of chemoattractant (top right). They secrete a
second chemoattractant (orange) and move up its gradient, which keeps the
group together. Model after Zhang, Chai, Sun and Lykotrafitis (2020).

## Fluids

![Two fluids that do not mix separate](assets/gallery/dpd_demixing.gif)

1200 particles of two species with dissipative particle dynamics: unlike
particles repel more strongly than like ones, and the mixture separates.

## Before and after training

Policies trained with the PPO baseline of the library
([Learning](learning.md)). Each animation shows the policy before training
on the left and after training on the right, from the same start, for the
seed with the median return of five.

![Swimmers before and after training](assets/gallery/learned_microswimmer.gif)

400 swimmers in the field of vortices; the picture shows 4 × 4 periods of the
flow and the altitude that the swimmers have reached. The trained swimmers
sense the vorticity and climb.

![The brick wall before and after training](assets/gallery/learned_brick_wall.gif)

![The room with four doors before and after training](assets/gallery/learned_evacuation.gif)

40 pedestrians in a room with four doors. The trained pedestrians leave, but
more slowly than pedestrians that walk to the nearest door: this policy uses
two of the four doors.

## Three dimensions

![Agents leave a 3D chamber](assets/gallery/evacuation_chamber_3d.gif)

60 agents leave a chamber of 10 m × 10 m × 4 m through a door (blue) and a
hatch in the ceiling (orange). They move freely in all three directions;
there is no gravity. Rendered with OVITO.

![Drones encircle a moving target](assets/gallery/encircle_3d.gif)

Eight drones climb to an orbit above a moving target and spread evenly around
it. This environment lives in a separate repository,
[drone-encircle](https://github.com/ZhaojieC/drone-encircle), and is built on
the physics core of ParticleGym.
