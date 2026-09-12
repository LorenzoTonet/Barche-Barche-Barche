import numpy as np
import math

def update_boat(state, action, dt, polar_diagram):
    """
    Placeholder molto basilare, solo per testare il loop step/reset.
    Pura funzione: non tocca state, restituisce solo i nuovi valori.
    Ignora vento, angolo vela, attrito: accelerazione fissa nella
    direzione in cui punta la barca.
    """
    boat_position = state["boat_position"]
    boat_velocity = state["boat_velocity"]
    boat_angle = state["boat_angle"]

    boat_rotation_intensity = float(action["boat_rotation"][0]) # [-1, 1]
    true_rotation_velocity = boat_rotation_intensity * boat_velocity 


    # compute difference between boat angle and wind angle
    raw_angle = boat_angle + true_rotation_velocity * dt
    new_boat_angle = np.arctan2(np.sin(raw_angle), np.cos(raw_angle))

    wind_angle = (math.atan2(state["wind_vector"][1], state["wind_vector"][0])) % (2 * np.pi) - np.pi
    angle_diff = abs(np.arctan2(np.sin(wind_angle - new_boat_angle), np.cos(wind_angle - new_boat_angle)))

    # compute the maximum velocity based on the angle difference
    max_velocity = polar_diagram(angle_diff) * (np.linalg.norm(state["wind_vector"]))

    # compute acceleration based on the maximum velocity and current velocity
    const = .02
    acceleration = const * (max_velocity**2 - boat_velocity**2)

    # compute the new velocity and position based on the acceleration and time step
    new_velocity = boat_velocity + acceleration * dt
    new_position_x = boat_position[0] + new_velocity * dt * math.cos(new_boat_angle)
    new_position_y = boat_position[1] + new_velocity * dt * math.sin(new_boat_angle)

    new_position = np.array([new_position_x, new_position_y])
    
    return {
        "velocity": new_velocity,
        "position": new_position,
        "boat_angle": new_boat_angle
    }