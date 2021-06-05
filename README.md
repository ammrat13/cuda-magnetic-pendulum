# CUDA Magnetic Pendulum

This code simulates (an approximation of) a magnetic pendulum - a classic
demonstration of chaos. It starts the pendulum off at different positions and
steps forward to determine where it ends up. It then generates an image with
each pixel colored depending on the result.

The simulation has a spring force toward the origin as well as two
inverse-square fields from plus/minus one-half on the y-axis. The user can't
change this field, however they can change the timestep, iterations, and
resolution of the simulation.

Credit to [Beltoforion](https://beltoforion.de/en/magnetic_pendulum/) for the
idea.
