# Barche-Barche-Barche
This repository contains the project for the final exam of the course "Reinforcement Learning 2026"

> *A sailboat or sailing boat is a boat propelled partly or entirely by sails and is smaller than a sailing ship. Distinctions in what constitutes a sailing boat and ship vary by region and maritime culture.*  
> — Wikipedia

The goal of this project is to produce an agent able to ride a sail boat to reach a **GOAL** while passing through a sequence of ordered **CHECKPOINTS** in the smallest time possible. The main difficulty of this task is that to move the boat its necessary to align the boat to a favorable angle in respect to the wind (In literature this is commonly adressed as [Zermelo's navigation problem](https://en.wikipedia.org/wiki/Zermelo%27s_navigation_problem)). 
Since both the state and the action spaces are continous, the main techniques used to optimize this kind of task involve Deep Neural Networks as function approximators for the value function and the policy. 

--- 
## 1. Action Space

Continuous action space consisting of a single control input:

| Action | Range | Description |
| :--- | :---: | :--- |
| **Rotation intensity** | `[-1.0, 1.0]` | Controls the steering / rotational force applied to the boat |

---

## 2. Observation Space

The observation space is a continuous vector representing the current state of the boat, the environment, and upcoming checkpoints:

| Variable | Type / Shape | Description |
| :--- | :---: | :--- |
| **Boat position** | `Vector2D` | Absolute position of the boat $(x, y)$ |
| **Boat speed** | `float` | Scalar magnitude of the boat velocity |
| **Boat angle** | `float` | Current heading angle of the boat |
| **Wind vector** | `Vector2D` | Local wind vector at current boat position $(w_x, w_y)$ |
| **Next checkpoint position** | `Vector2D` | Absolute coordinates of the next checkpoint $(x_{cp1}, y_{cp1})$ |
| **Next checkpoint relative** | `Vector3D` | Relative metrics to next checkpoint $[dx_{cp1}, dy_{cp1}, \text{distance}_{cp1}]$ |
| **Next-Next checkpoint position** | `Vector2D` | Absolute coordinates of the target after the next checkpoint $(x_{cp2}, y_{cp2})$ |
| **Next-Next checkpoint relative** | `Vector3D` | Relative metrics to next-next checkpoint $[dx_{cp2}, dy_{cp2}, \text{distance}_{cp2}]$ |

---

## 3. Reward Function

The reward function combines continuous potential-based reward shaping with sparse pointwise event signals:

### Continuous Signal
Rewards progress towards the next checkpoint while applying a constant penalty per time step to encourage efficiency:

```math
R(s, a, s') = (\phi(s) - \phi(s')) \cdot 150 - 0.01 + \text{bonus}
```

Where $\phi(s)$ represents the distance to the next checkpoint at state $s$. The scaling factor $150$ was tuned empirically during experimentation by computing the magnitudes of other terms and balancing it in respect to them. The last term "bonus" is simply an immediate reward for getting a checkpoint during the run that assumes values as:
```math
\text{bonus} = \begin{cases} +30 & \text{when getting a checkpoint} \\ 0 & \text{else} \end{cases}
```

### Terminal & Sparse Signals
Discrete rewards assigned upon specific environmental events:

```math
R(s, a, s') = \begin{cases} +50 & \text{if all checkpoints are reached} \\ -50 & \text{if the boat goes out of bounds} \end{cases}
```

---
## PHYSICS
The physics of the environment including the movement of the boat and the management of the wind is described in [EnvironmentPhysics.md](./EnvironmentPhysics.md)

---
## QUICK RESULTS

## REPOSITORY STRUCTURE

## HOW TO RUN
