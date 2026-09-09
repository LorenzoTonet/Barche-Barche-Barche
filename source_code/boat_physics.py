import numpy as np

def update_boat(state, action):
    """
    Placeholder molto basilare, solo per testare il loop step/reset.
    Pura funzione: non tocca state, restituisce solo i nuovi valori.
    Ignora vento, angolo vela, attrito: accelerazione fissa nella
    direzione in cui punta la barca.
    """
    boat_position = state["boat_position"]
    boat_velocity = state["boat_velocity"]
    boat_angle = state["boat_angle"]
    sail_angle = state["sail_angle"]

    sail_rotation = float(action["sail_rotation"][0])
    boat_rotation = float(action["boat_rotation"][0])

    new_boat_angle = boat_angle + boat_rotation * 0.1
    new_sail_angle = sail_angle + sail_rotation * 0.1

    forward_dir = np.array([np.cos(new_boat_angle), np.sin(new_boat_angle)])
    new_acceleration = forward_dir * .50  # valore fisso, solo per test

    new_velocity = boat_velocity + new_acceleration * 0.1  # dt fittizio
    new_position = boat_position + new_velocity * 0.1

    return {
        "acceleration": new_acceleration,
        "velocity": new_velocity,
        "position": new_position,
        "boat_angle": new_boat_angle,
        "sail_angle": new_sail_angle,
    }