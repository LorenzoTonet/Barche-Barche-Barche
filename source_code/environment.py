import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from source_code.boat_physics import update_boat
from source_code.vector_field import VecField
from source_code.map_elements import Checkpoint


class SailingEnv(gym.Env):

    def __init__(self, config: dict, wind_vec_field: VecField, goal: Checkpoint, checkpoints: list, render_mode: str = None):
        self.config = config
        self.checkpoints = checkpoints
        self.n_checkpoints = len(checkpoints)
        self.goal = goal

        self.wind_vec_field = wind_vec_field

        self.polar_diagram = self.create_polar_diagram(config, config["polar_diagram_vals"])

        self.steps = 0
        self.max_steps = config["max_steps"]
        self.max_speed = config["max_speed"]
        self.friction_coefficient = config["water_friction"]
        self.boat_rotation_speed = config["boat_rotation_speed"]
        self.dt = config["dt"]

        self.terminated = False
        self.truncated = False

        #ACTIONS = ROTATE_LEFT_SAIL, ROTATE_RIGHT_SAIL, ROTATE_LEFT_BOAT, ROTATE_RIGHT_BOAT
        # The action space is a continuous 2D vector representing the rotation angle of the sail and the boat
        # for simplicity it will be parameterized as a 2D vector with values in the range [-1, 1] for both dimensions
        self.action_space = spaces.Dict({
            "boat_rotation": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        # OBSERVATIONS = 
        self.observation_space = spaces.Dict({
            "boat_position": spaces.Box(low=np.array([0, 0]), high=np.array([config["map_width"], config["map_height"]]), dtype=np.float32),
            "boat_velocity": spaces.Box(low=(-np.inf), high=np.array(np.inf), dtype=np.float32),
            "boat_angle": spaces.Box(low=-np.pi, high=np.pi, dtype=np.float32),
            "wind_vector": spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), dtype=np.float32),
            "next_checkpoint_relative": spaces.Box(low=np.array([0, 0]), high=np.array([config["map_width"], config["map_height"]]), dtype=np.float32),
            "next_next_checkpoint_relative": spaces.Box(low=np.array([0, 0]), high=np.array([config["map_width"], config["map_height"]]), dtype=np.float32),
            "goal_relative": spaces.Box(low=np.array([0, 0]), high=np.array([config["map_width"], config["map_height"]]), dtype=np.float32)
        })

        self.state = self._create_initial_state()

        self.render_mode = render_mode
        self.window_size = (config["window_width"], config["window_height"])
        self.window = None
        self.clock = None


    def _calc_relative_dist_(self, point: Checkpoint):
        boat_x = self.state["boat_position"][0]
        boat_y = self.state["boat_position"][1]
        point_x, point_y = point.position

        dx = point_x - boat_x
        dy = point_y - boat_y
        distance = np.linalg.norm(self.state["boat_position"] - point.position)

        return np.array([dx, dy, distance])

    def _get_observation(self):
        boat_position = self.state["boat_position"]
        boat_velocity = self.state["boat_velocity"]
        boat_angle = self.state["boat_angle"]
        wind_vector = self.state["wind_vector"]
        next_checkpoint_idx = self.state["next_checkpoint_idx"]

        if next_checkpoint_idx < self.n_checkpoints:
            next_checkpoint = self.checkpoints[next_checkpoint_idx]
            next_next_checkpoint = self.checkpoints[next_checkpoint_idx + 1] if next_checkpoint_idx + 1 < self.n_checkpoints else None
        else:
            next_checkpoint = None
            next_next_checkpoint = None

        observation = {
            "boat_position": boat_position,
            "boat_velocity": boat_velocity,
            "boat_angle": boat_angle,
            "wind_vector": wind_vector,
            "next_checkpoint_relative": self._calc_relative_dist_(next_checkpoint) if next_checkpoint else np.array([0, 0, 0]),
            "next_next_checkpoint_relative": self._calc_relative_dist_(next_next_checkpoint) if next_next_checkpoint else np.array([0, 0, 0]),
            "goal_relative": self._calc_relative_dist_(self.goal)
        }

        return observation

    def _create_initial_state(self):
        return {
            "boat_position": self.config["initial_position"].copy(),
            "boat_velocity": 0.,
            "boat_angle": self.config["initial_boat_angle"],
            "wind_vector": self.wind_vec_field.get_vec(self.config["initial_position"]),
            "visited_checkpoints": [False] * self.n_checkpoints,
            "next_checkpoint_idx": 0,
        }

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = self._create_initial_state()
        self.steps = 0
        return self._get_observation(), {}

    def create_polar_diagram(self, config, polar_diagram_vals):
        """
        Create a polar diagram (i.e. a function that maps angles to speeds) from the given values.
        """
        angles_deg, raw_values = zip(*polar_diagram_vals)
        angles_deg = np.array(angles_deg)
        raw_values = np.array(raw_values)

        # scale values to [0, 1]
        max_val = np.max(raw_values)
        values = raw_values / max_val

        # convert angles to radians for spline fitting
        angles_rad = np.radians(angles_deg)

        # interpolate
        cs = CubicSpline(angles_rad, values, bc_type=((2, 0), (1, 0)))

        if config.get("plot", False):
            fine_angles_deg = np.linspace(0, 180, 200)
            fine_angles_rad = np.radians(fine_angles_deg)
            fine_values = cs(fine_angles_rad)

            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.plot(angles_deg, values, "ro", label="Data points")
            plt.plot(fine_angles_deg, fine_values, "b-", label="Cubic Spline")
            plt.title("Boat Speed vs True Wind Angle (TWA)")
            plt.xlabel("TWA (°)")
            plt.ylabel("Normalized Boat Speed")
            plt.grid(True)
            plt.legend()

            plt.subplot(1, 2, 2, projection="polar")
            plt.gca().set_theta_zero_location("N")
            plt.gca().set_theta_direction(-1)  # Clockwise
            plt.plot(fine_angles_rad, fine_values, "b-", label="Starboard")
            plt.plot(-fine_angles_rad, fine_values, "b--", label="Port (Symmetric)")
            plt.title("Polar Diagram", y=1.08)
            plt.grid(True)

            plt.tight_layout()
            plt.savefig("polar_plot.png", dpi=150)
            plt.close()

        return cs

    def step(self, action):
        # Placeholder

        # For reference: 
        # velocity = velocity + forward direction * acceleration
        # position = position + velocity * dt
        # The forward direction is determined by the boat's angle
        # The acceleration is determined by the wind force on the sail, which is a function of the wind vector and the sail angle

        update = update_boat(self.state, action, self.dt, self.polar_diagram)

        self.state["boat_position"] = update["position"]
        self.state["boat_velocity"] = update["velocity"]
        self.state["boat_angle"] = update["boat_angle"]
        self.state["wind_vector"] = self.wind_vec_field.get_vec(self.state["boat_position"])
        print(f" Wind vector: {self.state['wind_vector']}")

        reward = self.reward_function()

        self.steps += 1

        terminated = False
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True
            terminated = True

        info = {}
        return self._get_observation(), reward, terminated, truncated, info

    def reward_function(self):
        # Placeholder reward function
        # For now, let's just give a reward of 1 for each step the boat is moving towards the goal
        boat_position = self.state["boat_position"]
        goal_position = self.goal.position

        distance_to_goal = np.linalg.norm(boat_position - goal_position)

        # Reward is inversely proportional to the distance to the goal
        reward = 1.0 / (distance_to_goal + 1e-5)  # Add a small value to avoid division by zero

        return reward


    # Rendering functions (from Claude)
    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Sailing Env")
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((10, 60, 120))  # water

        scale_x = self.window_size[0] / self.config["map_width"]
        scale_y = self.window_size[1] / self.config["map_height"]

        def to_screen(pos):
            # flip y: nel mondo l'asse y punta in alto, in pygame in basso
            return int(pos[0] * scale_x), int(self.window_size[1] - pos[1] * scale_y)

        # checkpoints
        for i, cp in enumerate(self.checkpoints):
            visited = self.state["visited_checkpoints"][i]
            color = (0, 200, 0) if visited else (220, 200, 0)
            pygame.draw.circle(canvas, color, to_screen(cp.position), 6)

        # goal
        pygame.draw.circle(canvas, (220, 0, 0), to_screen(self.goal.position), 8)

        # origin
        pygame.draw.circle(canvas, (220, 0, 0), to_screen((0, 0)), 8)

        # boat (triangolino orientato secondo boat_angle)
        bx, by = to_screen(self.state["boat_position"])
        angle = float(self.state["boat_angle"])
        size = 10
        p1 = (bx + size * np.cos(angle), by - size * np.sin(angle))
        p2 = (bx + size * np.cos(angle + 2.5), by - size * np.sin(angle + 2.5))
        p3 = (bx + size * np.cos(angle - 2.5), by - size * np.sin(angle - 2.5))
        pygame.draw.polygon(canvas, (255, 255, 255), [p1, p2, p3])

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.config["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None