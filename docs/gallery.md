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
