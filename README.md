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
The boat dynamics are modeled using a simplified deterministic physics model. At each time step, the boat's heading is updated according to the action, then the maximum achievable speed is computed from the wind direction and the polar diagram.

### 1. Boat Rotation
The action controls the boat's rotation intensity where:
- $a \in [-1,1]$ is the input action.
- $r$ is the rotation intensity ($r = a$).

The actual angular velocity depends on the current boat speed to simulate the fact that if the boat is not moving, rotating it is difficult:
```math
\omega = r \cdot v
```
where $v$ is the current boat speed.

The heading is then updated over a time step $\Delta t$:

```math
\theta_{\text{raw}} =
\theta + \omega \Delta t
```

The resulting angle is normalized to the interval $[-\pi,\pi]$

### 2. Relative Wind Angle
The wind direction is obtained from the wind vector

```math
\mathbf{w} =
\begin{bmatrix}
w_x\\
w_y
\end{bmatrix}
```
as
```math
\theta_w =
atan2(w_y,w_x)
```

The relative angle between the boat heading and the wind direction is computed as

```math
\Delta\theta =
\left|
\operatorname{atan2}
\left(
\sin(\theta_w-\theta'),
\cos(\theta_w-\theta')
\right)
\right|
```

This gives the smallest angular difference between the boat's heading and the wind direction, with

```math
\Delta\theta \in [0,\pi].
```

### 3. Maximum Boat Speed

The polar diagram describes the maximum boat speed as a function of the relative wind angle.

The maximum achievable speed is calculated as

```math
v_{\max} =
P(\Delta\theta)\,\|\mathbf{w}\|
```

where:

- $P(\Delta\theta)$ is the polar diagram function.
- $\|\mathbf{w}\|$ is the wind speed:

```math
\|\mathbf{w}\| =
\sqrt{w_x^2+w_y^2}
```

Thus, the maximum boat speed depends both on the wind speed and on the angle at which the boat is sailing relative to the wind.

### 4. Acceleration

The boat accelerates towards its maximum achievable speed according to

```math
a_v =
k
\left(
v_{\max}^2-v^2
\right)
```
where $k$ is the acceleration constant. The new speed is then obtained using Euler integration:

```math
v' = v+a_v\Delta t
```

or, equivalently,

```math
v' =
v+
k
\left(
v_{\max}^2-v^2
\right)
\Delta t.
```

### 5. Position Update

The boat moves in the direction of its new heading $\theta'$.

The new position is

```math
x' = x+v'\Delta t\cos(\theta')
```

```math
y' = y+v'\Delta t\sin(\theta')
```
---

---
## QUICK RESULTS

## REPOSITORY STRUCTURE

## HOW TO RUN
