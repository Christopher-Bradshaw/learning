# Theory

## Direct summation

The simplest way to write an N-body sim, is to just compute the force on each particle, by each particle. This takes n^2 operations per time step though, and so is impractical.

Note that we need a softening length to avoid forces going to infinity if the particles are at the same location. This is physically motivated as the particles are representative of an extended mass distribution.

## Particle mesh
