import colorsys
import math
import imageio
import numpy as np
import pygame

"""
This file contains the code to make a video with multiple boats running simultaneously on the same environment.
It is entirely made by an LLM.
"""

def generate_distinct_colors(n):
    """Generates N distinct bright RGB colors for rendering boats."""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    return colors


def render_multi_boats(cfg, env, agent):
    n_episodes = cfg['evaluate']['n_episodes']
    output_path = cfg['evaluate'].get('video_path', 'multi_run_simultaneous.mp4')
    fps = cfg.get('render_fps', 30)

    base_env = env.unwrapped
    trajectories = []

    print(f"Simulating {n_episodes} runs...")

    # Phase 1: Collect trajectory state data for each run
    for i in range(n_episodes):
        print(f"Simulating episode {i+1}/{n_episodes}")
        # Use same seed across runs if you want identical map/start conditions
        state, _ = env.reset(seed=cfg['evaluate'].get('seed', 42) + i if 'seed' in cfg['evaluate'] else None)
        done = truncated = False
        
        run_data = []
        while not (done or truncated):
            run_data.append({
                "position": np.copy(base_env.state["boat_position"]),
                "angle": float(base_env.state["boat_angle"]),
                "speed": float(base_env.state["boat_speed"]),
                "wind_vector": np.copy(base_env.state["wind_vector"]),
                "visited": list(base_env.state["visited_checkpoints"]),
            })

            action, _ = agent.get_action(state, deterministic=True)
            state, reward, done, truncated, info = env.step(action)

        # Store final terminal step
        run_data.append({
            "position": np.copy(base_env.state["boat_position"]),
            "angle": float(base_env.state["boat_angle"]),
            "speed": float(base_env.state["boat_speed"]),
            "wind_vector": np.copy(base_env.state["wind_vector"]),
            "visited": list(base_env.state["visited_checkpoints"]),
        })

        trajectories.append(run_data)

    # Phase 2: Render frame-by-frame up to the longest episode length
    max_steps = max(len(traj) for traj in trajectories)
    print(f"Longest episode finished in {max_steps} steps. Generating video...")

    window_size = base_env.window_size
    map_width = base_env.config["map_width"]
    map_height = base_env.config["map_height"]
    checkpoints = base_env.checkpoints

    scale_x = window_size[0] / map_width
    scale_y = window_size[1] / map_height

    def to_screen(pos):
        return int(pos[0] * scale_x), int(window_size[1] - pos[1] * scale_y)

    boat_colors = generate_distinct_colors(n_episodes)

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 16)
    canvas = pygame.Surface(window_size)

    frames = []

    for t in range(max_steps):
        canvas.fill((10, 60, 120))  # Water background

        # Draw checkpoints
        for i_cp, cp in enumerate(checkpoints):
            pygame.draw.circle(canvas, (220, 200, 0), to_screen(cp.position), 6)
            if i_cp == len(checkpoints) - 1:
                pygame.draw.circle(canvas, (255, 0, 0), to_screen(cp.position), int(cp.radius * scale_x), 2)

        # Draw origin
        pygame.draw.circle(canvas, (220, 0, 0), to_screen((0, 0)), 8)

        active_boats = 0

        # Draw trajectory lines and boat icons for each episode
        for ep_idx, traj in enumerate(trajectories):
            step_idx = min(t, len(traj) - 1)
            is_active = t < (len(traj) - 1)
            if is_active:
                active_boats += 1

            curr_state = traj[step_idx]
            color = boat_colors[ep_idx]

            # 1. Trajectory path up to step t
            path_points = [to_screen(snap["position"]) for snap in traj[:step_idx + 1]]
            if len(path_points) > 1:
                line_color = color if is_active else tuple(c // 2 for c in color)
                pygame.draw.lines(canvas, line_color, False, path_points, 2 if is_active else 1)

            # 2. Boat triangle
            bx, by = to_screen(curr_state["position"])
            angle = curr_state["angle"]
            size = 10 if is_active else 6
            boat_color = color if is_active else tuple(c // 2 for c in color)

            p1 = (bx + size * np.cos(angle), by - size * np.sin(angle))
            p2 = (bx + size * np.cos(angle + 2.5), by - size * np.sin(angle + 2.5))
            p3 = (bx + size * np.cos(angle - 2.5), by - size * np.sin(angle - 2.5))
            pygame.draw.polygon(canvas, boat_color, [p1, p2, p3])

        # Wind & HUD details (ref derived from run 0)
        ref_snap = trajectories[0][min(t, len(trajectories[0]) - 1)]
        wind_vector = ref_snap["wind_vector"]
        wind_speed = float(np.linalg.norm(wind_vector))
        wind_dir = math.atan2(wind_vector[1], wind_vector[0])

        compass_center = (60, 60)
        compass_radius = 40
        pygame.draw.circle(canvas, (230, 230, 230), compass_center, compass_radius, 1)

        arrow_end = (
            compass_center[0] + compass_radius * math.cos(wind_dir),
            compass_center[1] - compass_radius * math.sin(wind_dir),
        )
        pygame.draw.line(canvas, (255, 220, 0), compass_center, arrow_end, 3)

        hud_lines = [
            f"Step: {t}/{max_steps}",
            f"Active boats: {active_boats}/{n_episodes}",
            f"Wind speed: {wind_speed:.2f}"
        ]
        for l_idx, line in enumerate(hud_lines):
            surf = font.render(line, True, (255, 255, 255))
            canvas.blit(surf, (10, 110 + l_idx * 18))

        # Convert canvas to image frame (H, W, 3)
        frame = np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))
        frames.append(frame)

    print(f"Saving video ({len(frames)} frames) to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print("Video saved successfully!")
