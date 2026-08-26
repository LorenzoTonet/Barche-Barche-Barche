import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Config:
    # Boat configuration parameters
    max_speed: float = 10.0
    sail_rotation_speed: float = 0.1
    boat_rotation_speed: float = 0.05
    initial_sail_angle: float = 0.0
    initial_boat_angle: float = 0.0
    initial_position: np.ndarray = np.array([0.0, 0.0])

    # Map configuration parameters
    map_width: int = 100
    map_height: int = 100
    water_friction: float = 0.1

    # Simulation parameters
    dt: float = 0.1
    max_steps: int = 200


class Checkpoint:
    def __init__(self, position, number, radius):
        self.position = position
        self.number = number
        self.radius = radius


class VecField():
    def __init__(self, space_length, space_width, function):
        pass
    def get_vec(self, point2d):
        return

class SailingEnv(gym.Env):

    def __init__(self, config: Config, wind_vec_field: VecField, goal: Checkpoint, checkpoints: list):
        self.config = config
        self.checkpoints = checkpoints
        self.n_checkpoints = len(checkpoints)
        self.goal = goal

        self.wind_vec_field = wind_vec_field

        self.steps = 0
        self.max_steps = config.max_steps
        self.max_speed = config.max_speed
        self.friction_coefficient = config.water_friction
        self.sail_rotation_speed = config.sail_rotation_speed
        self.boat_rotation_speed = config.boat_rotation_speed
        self.dt = config.dt

        self.terminated = False
        self.truncated = False

        #ACTIONS = ROTATE_LEFT_SAIL, ROTATE_RIGHT_SAIL, ROTATE_LEFT_BOAT, ROTATE_RIGHT_BOAT
        # The action space is a continuous 2D vector representing the rotation angle of the sail and the boat
        # for simplicity it will be parameterized as a 2D vector with values in the range [-1, 1] for both dimensions
        self.action_space = spaces.Dict({
            "sail_rotation": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "boat_rotation": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        # OBSERVATIONS = 
        self.observation_space = spaces.Dict({
            "boat_position": spaces.Box(low=np.array([0, 0]), high=np.array([config.map_width, config.map_height]), dtype=np.float32),
            "boat_velocity": spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), dtype=np.float32),
            "sail_angle": spaces.Box(low=-np.pi, high=np.pi, dtype=np.float32),
            "boat_angle": spaces.Box(low=-np.pi, high=np.pi, dtype=np.float32),
            "wind_vector": spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), dtype=np.float32),
            "next_checkpoint_relative": spaces.Box(low=np.array([0, 0]), high=np.array([config.map_width, config.map_height]), dtype=np.float32),
            "next_next_checkpoint_relative": spaces.Box(low=np.array([0, 0]), high=np.array([config.map_width, config.map_height]), dtype=np.float32),
            "goal_relative": spaces.Box(low=np.array([0, 0]), high=np.array([config.map_width, config.map_height]), dtype=np.float32)
        })

        self.state = self._create_initial_state()


    def _calc_relative_dist_(self, point: Checkpoint):
        boat_x, boat_y = self.state["boat_position"]
        point_x, point_y = point.position
    
        dx = point_x - boat_x
        dy = point_y - boat_y
        distance = np.linalg.norm(self.state["boat_position"] - point.position)

        return np.array([dx, dy, distance])

    def _get_observation(self):
        boat_position = self.state["boat_position"]
        boat_velocity = self.state["boat_velocity"]
        sail_angle = self.state["sail_angle"]
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
            "sail_angle": sail_angle,
            "boat_angle": boat_angle,
            "wind_vector": wind_vector,
            "next_checkpoint_relative": self._calc_relative_dist_(next_checkpoint) if next_checkpoint else np.array([0, 0, 0]),
            "next_next_checkpoint_relative": self._calc_relative_dist_(next_next_checkpoint) if next_next_checkpoint else np.array([0, 0, 0]),
            "goal_relative": self._calc_relative_dist_(self.goal)
        }

        return observation

    def _create_initial_state(self):
        return {
            "boat_position": self.config.initial_position.copy(),
            "boat_velocity": np.array([0., 0.]),
            "sail_angle": self.config.initial_sail_angle,
            "boat_angle": self.config.initial_boat_angle,
            "wind_vector": self.wind_vec_field.get_vec(self.config.initial_position),
            "visited_checkpoints": [False] * self.n_checkpoints,
            "next_checkpoint_idx": 0,
        }

    def _calculate_velocity(self, wind_force, boat_angle):
            # Placeholder
            velocity = 12
            return velocity

    def _calculate_wind_force(self):
        # Placeholder
        wind_velocity = self.state["wind_vector"]
        sail_angle = self.state["sail_angle"]
        # Calculate the wind force based on the wind velocity and sail angle
        wind_force = np.array([wind_velocity[0] * np.cos(sail_angle), wind_velocity[1] * np.sin(sail_angle)])
        return wind_force

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = self._create_initial_state()
        self.steps = 0
        return self._get_observation(), {}

    def step(self, action):

        state = self.state
        new_sail_angle = state["sail_angle"] + action["sail_rotation"][0] * self.sail_rotation_speed * self.dt
        new_boat_angle = state["boat_angle"] + action["boat_rotation"][0] * self.boat_rotation_speed * self.dt
        self.state["sail_angle"] = np.clip(new_sail_angle, -np.pi, np.pi)
        self.state["boat_angle"] = np.clip(new_boat_angle, -np.pi, np.pi)

        wind_force = self._calculate_wind_force()

        self.state["boat_velocity"] = self.state["boat_velocity"] + wind_force * self.dt
        self.state["boat_position"] = self.state["boat_position"] + self.state["boat_velocity"]

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
        pass