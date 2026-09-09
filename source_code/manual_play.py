import numpy as np
import pygame
from environment import SailingEnv, Config
from vector_field import VecField
from map_elements import Checkpoint


def play_manual():
    config = Config()
    wind_vec_field = VecField(space_height = config.map_height, space_width = config.map_width, function = None)
    goal = Checkpoint(position=np.array([50.0, 50.0]), number=1, radius=5.0)
    checkpoints = []

    env = SailingEnv(config, wind_vec_field, goal, checkpoints, render_mode="human")
    obs, info = env.reset()
    env.render()

    action = {
        "sail_rotation": np.array([0.0], dtype=np.float32),
        "boat_rotation": np.array([0.0], dtype=np.float32),
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        boat_rotation = 0.0
        sail_rotation = 0.0
        if keys[pygame.K_LEFT]:
            boat_rotation = -1.0
        if keys[pygame.K_RIGHT]:
            boat_rotation = 1.0
        if keys[pygame.K_UP]:
            sail_rotation = 1.0
        if keys[pygame.K_DOWN]:
            sail_rotation = -1.0

        action["boat_rotation"][0] = boat_rotation
        action["sail_rotation"][0] = sail_rotation

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    play_manual()