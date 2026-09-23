# Barche-Barche-Barche
This repository contains the project for the final exam of the course "Reinforcement Learning 2026"

> *A sailboat or sailing boat is a boat propelled partly or entirely by sails and is smaller than a sailing ship. Distinctions in what constitutes a sailing boat and ship vary by region and maritime culture.*  
> — Wikipedia

The goal of this project is to produce an agent able to ride a sail boat to reach a **GOAL** while passing through a sequence of ordered **CHECKPOINTS** in the smallest time possible. The main difficulty of this task is that to move the boat its necessary to align the boat to a favorable angle in respect to the wind (In literature this is commonly adressed as [Zermelo's navigation problem](https://en.wikipedia.org/wiki/Zermelo%27s_navigation_problem)). 
Since both the state and the action spaces are continous, the main techniques used to optimize this kind of task involve Deep Neural Networks as function approximators for the value function and the policy. 

--- 
## SPECIFICATIONS OF THE ENVIRONMENT
- action space
- observtions space
### ACTION SPACE
- "rotation intensity" [-1, 1]
  
### OBSERVATION SPACE
- Boat position (vector 2D)
- Boat speed (float)
- Boat angle (float)
- Wind vector in current position (vector 2D)
- Next checkpoint position (vector 2D)
- Next checkpoint relatives [distance_x,distance_y, distance] (vector 3D)
- Next-Next checkpoint position (vector 2D)
- Next-Next checkpoint relatives [distance_x,distance_y, distance] (vector 3D)

REWARD FUNCTION
The reward shaping for the actual environment has two different kind of signals for the agent:
- A continous signal that reward the aproaching to a checkpoint and at the same time penalize the absolute number of steps 
```math
R(s, a, s') = (\phi(s) - \phi(s')) * 150 - 0.01 + bonus
bonus = 
```
In this context, the function $\phi(s)$ is the distance from the next checkpoint in state s and it embodies a measure of progress. The coefficent associated is tuned during the process of experimentation.
- Some "pointwise" signals when the goal is reached or the boat goes out of bounds
```math
R(s, a, s') = 50 \quad \text{if got all checkpoints}
```
```math
R(s, a, s') = -50 \quad \text{if the agent goes out of bounds}
```
---
## PHYSICS
- environment
- boat movement formule

---
## QUICK RESULTS

## REPOSITORY STRUCTURE

## HOW TO RUN
