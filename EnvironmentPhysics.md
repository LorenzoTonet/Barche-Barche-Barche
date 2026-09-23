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
atan2
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