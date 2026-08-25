# Barche-Barche-Barche
This repository contains the project for the final exam of the course "Reinforcement Learning"

> *A sailboat or sailing boat is a boat propelled partly or entirely by sails and is smaller than a sailing ship. Distinctions in what constitutes a sailing boat and ship vary by region and maritime culture.*  
> — Wikipedia

The goal of the task is to reach a **GOAL** while passing through a sequence of **CHECKPOINTS**.

The goal of the task is to reach a GOAL by passing through some checkpoints.
---
## SPECIFICATIONS
STATE space = [position2d_boat, position2d_goal, position2d_CP1, ..., position2d_CPn, boat_rotation, sail_rotation, rudder_rotation, boat_velocity, wind_vectorial_field, collected_CPs]

ACTION space = [rotate_sail, rotate_rudder]

OBSERVATION space = [full_observability]

---
## MOVEMENT PHYSICS

TODO

---
## FRAMEWORK
This project will rely on gymnasium environment due to its standardized interface and facility of use.
